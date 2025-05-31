import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from fresunet import FresUNet
from dataset import ChangeDetectionDataset
from scipy.ndimage import binary_opening, binary_closing
import torch.nn as nn
import torch.nn.functional as F
import cv2

TRAIN_CITIES = [
    'aguasclaras','bercy','bordeaux','nantes','paris','rennes','saclay_e',
    'abudhabi','cupertino','pisa','beihai','hongkong','beirut','mumbai'
]
TEST_CITIES = [
    'brasilia','montpellier','norcia','rio','saclay_w','valencia',
    'dubai','lasvegas','milano','chongqing'
]

def custom_collate(batch):
    img1s = torch.stack([item[0] for item in batch])
    img2s = torch.stack([item[1] for item in batch])
    masks = torch.stack([item[2] for item in batch])
    time_diffs = torch.stack([item[3] for item in batch])
    return img1s, img2s, masks, time_diffs

def compute_metrics_np(pred, gt):
    tn, fp, fn, tp = confusion_matrix(gt.flatten(), pred.flatten(), labels=[0,1]).ravel()
    p = tp/(tp+fp) if tp+fp > 0 else 0
    r = tp/(tp+fn) if tp+fn > 0 else 0
    i = tp/(tp+fp+fn) if tp+fp+fn > 0 else 0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn > 0 else 0
    oa = (tp+tn)/(tp+tn+fp+fn)
    return p, r, i, f1, oa

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = nn.BCEWithLogitsLoss()(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def apply_guided_filter(probs, img, radius=5, eps=0.1):
    h, w = probs.shape[1:]
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)
    if img_np.shape[2] == 13:
        img_np = img_np[:, :, [3, 2, 1]]
    
    prob_foreground = probs[1].cpu().numpy()
    prob_foreground = (prob_foreground * 255).astype(np.uint8)
    
    guided = cv2.ximgproc.guidedFilter(guide=img_np, src=prob_foreground, radius=radius, eps=eps)
    guided = guided.astype(np.float32) / 255.0
    return (guided > 0.5).astype(np.uint8)

def evaluate_model(nets, loader, device, desc="Eval", use_guided_filter=False, use_all_bands=False):
    for net in nets:
        net.eval()
    all_metrics = []
    best_threshold = 0.7
    best_f1 = 0
    
    print(f"[INFO] {desc} on {len(loader)} samples")
    
    for idx, (im1, im2, gt, time_diff) in enumerate(loader):
        im1, im2, time_diff = im1.to(device), im2.to(device), time_diff.to(device)
        ensemble_probs = []
        
        with torch.no_grad():
            for net in nets:
                out, _ = net(im1, im2, time_diff)
                probs = torch.softmax(out, dim=1)
                ensemble_probs.append(probs[:, 1, :, :])
            change_prob = torch.mean(torch.stack(ensemble_probs), dim=0)
            
            thresholds = np.arange(0.1, 1.0, 0.1)
            metrics = []
            for thresh in thresholds:
                pred = (change_prob > thresh).long()
                p_np = pred.cpu().numpy().squeeze().astype(np.uint8)
                gt_np = gt.numpy().squeeze().astype(np.uint8)
                m = compute_metrics_np(p_np, gt_np)
                metrics.append(m)
                if m[3] > best_f1:
                    best_f1 = m[3]
                    best_threshold = thresh
            
            pred = (change_prob > best_threshold).long()
            p_np = pred.cpu().numpy().squeeze().astype(np.uint8)
            
            p_np = binary_opening(p_np, structure=np.ones((3,3))).astype(np.uint8)
            p_np = binary_closing(p_np, structure=np.ones((3,3))).astype(np.uint8)
            
            if use_guided_filter:
                probs_np = torch.softmax(out, dim=1).cpu().numpy().squeeze()
                p_np = apply_guided_filter(probs_np, im1[0], radius=5, eps=0.1)
            
            gt_np = gt.numpy().squeeze().astype(np.uint8)
            m = compute_metrics_np(p_np, gt_np)
            city = TRAIN_CITIES[idx % len(TRAIN_CITIES)]
            print(f"{city}: P={m[0]:.3f}, R={m[1]:.3f}, IoU={m[2]:.3f}, F1={m[3]:.3f}, OA={m[4]:.3f}, Threshold={best_threshold:.2f}")
            all_metrics.append(m)

    avg = np.mean(all_metrics, axis=0)
    print(f"{desc} Average → P={avg[0]:.3f}, R={avg[1]:.3f}, IoU={avg[2]:.3f}, F1={avg[3]:.3f}, OA={avg[4]:.3f}")
    return avg

def dice_loss(pred, target, smooth=1.0):
    pred = pred.sigmoid()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def combined_loss(pred, target, recon, x1, pos_weight, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    focal = FocalLoss(gamma=2.0)
    pred_prob = torch.softmax(pred, dim=1)[:, 1, :, :]
    target_binary = target.squeeze(1).float() if target.dim() == 4 else target.float()
    bce_loss = bce(pred_prob, target_binary)
    dice = dice_loss(pred_prob, target_binary)
    focal_loss = focal(pred_prob, target_binary)
    recon_loss = nn.MSELoss()(recon, x1)
    return alpha * bce_loss + beta * dice + delta * focal_loss + gamma * recon_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Path to data/')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval-train', action='store_true', help='Evaluate on TRAIN_CITIES')
    parser.add_argument('--eval-val', action='store_true', help='Evaluate on validation cities')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--use-all-bands', action='store_true', help='Use all 13 Sentinel-2 bands')
    parser.add_argument('--use-guided-filter', action='store_true', help='Use guided filter post-processing')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_nbr = 26 if args.use_all_bands else 6
    nets = [
        FresUNet(input_nbr=input_nbr, label_nbr=2).to(device),
        FresUNet(input_nbr=input_nbr, label_nbr=2).to(device)
    ]

    pretrained_paths = ['fresunet3_final.pth.tar', 'checkpoints/fresunet_best.pth']
    for i, net in enumerate(nets):
        pretrained_path = pretrained_paths[i] if i < len(pretrained_paths) and os.path.exists(pretrained_paths[i]) else None
        if pretrained_path:
            print(f"[INFO] Loading pre-trained model {i+1} from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
            print(f"[INFO] Loaded pre-trained weights for model {i+1}")

    if args.train:
        train_ds = ChangeDetectionDataset(args.root, TRAIN_CITIES[:10], augment=True, 
                                        require_mask=True, return_time=True, use_all_bands=args.use_all_bands)
        val_ds = ChangeDetectionDataset(args.root, TRAIN_CITIES[10:], augment=False, 
                                      require_mask=True, return_time=True, use_all_bands=args.use_all_bands)
        print(f"[INFO] Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                                num_workers=2, collate_fn=custom_collate)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                              num_workers=2, collate_fn=custom_collate)

        optimizers = [torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5) for net in nets]
        schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', 
                                                               factor=0.5, patience=3, min_lr=1e-6) 
                     for opt in optimizers]
        
        pw = train_ds.get_pos_weight() * 6.0
        early_stopping = EarlyStopping(patience=args.patience)

        best_val_f1 = 0
        for ep in range(args.epochs):
            for net, optimizer in zip(nets, optimizers):
                net.train()
                running_loss = 0.0
                for im1, im2, m, time_diff in train_loader:
                    im1, im2, m, time_diff = im1.to(device), im2.to(device), m.to(device), time_diff.to(device)
                    optimizer.zero_grad()
                    out, recon = net(im1, im2, time_diff)
                    loss = combined_loss(out, m, recon, im1, pos_weight=torch.tensor(pw, device=device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()
                    running_loss += loss.item()
                
                train_loss = running_loss / len(train_loader)
                print(f"Epoch {ep+1}/{args.epochs} Model {nets.index(net)+1} Loss: {train_loss:.4f}")

            val_metrics = evaluate_model(nets, val_loader, device, "Validation", 
                                       use_guided_filter=args.use_guided_filter, use_all_bands=args.use_all_bands)
            val_f1 = val_metrics[3]
            for scheduler in schedulers:
                scheduler.step(val_f1)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                os.makedirs('checkpoints', exist_ok=True)
                for i, net in enumerate(nets):
                    torch.save(net.state_dict(), f'checkpoints/fresunet_best_{i}.pth')
                print(f"[INFO] New best models saved (F1: {val_f1:.3f})")
            
            early_stopping(1 - val_f1)
            if early_stopping.early_stop:
                print("[INFO] Early stopping triggered")
                break

        print("[INFO] Training completed")

    elif args.eval_train:
        for i, net in enumerate(nets):
            ckpt = f'checkpoints/fresunet_best_{i}.pth'
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
            net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        
        ds = ChangeDetectionDataset(args.root, TRAIN_CITIES, augment=False, 
                                  require_mask=True, return_time=True, use_all_bands=args.use_all_bands)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate)
        evaluate_model(nets, loader, device, "Train Eval", use_guided_filter=args.use_guided_filter, use_all_bands=args.use_all_bands)

    elif args.eval_val:
        val_cities = TRAIN_CITIES[10:]
        ds = ChangeDetectionDataset(args.root, val_cities, augment=False, 
                                  require_mask=True, return_time=True, use_all_bands=args.use_all_bands)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate)
        
        for i, net in enumerate(nets):
            checkpoint_path = f'checkpoints/fresunet_best_{i}.pth'
            if os.path.exists(checkpoint_path):
                print(f"[INFO] Loading trained model {i+1} from {checkpoint_path}")
                net.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        
        print("\n[INFO] Validation Results:")
        all_metrics = []
        best_f1 = -1
        best_city = None
        best_pred = None
        best_gt = None
        
        for idx, (im1, im2, gt, time_diff) in enumerate(loader):
            im1, im2, time_diff = im1.to(device), im2.to(device), time_diff.to(device)
            ensemble_probs = []
            
            with torch.no_grad():
                for net in nets:
                    out, _ = net(im1, im2, time_diff)
                    probs = torch.softmax(out, dim=1)
                    ensemble_probs.append(probs[:, 1, :, :])
                change_prob = torch.mean(torch.stack(ensemble_probs), dim=0)
                
                thresholds = np.arange(0.1, 1.0, 0.1)
                metrics = []
                for thresh in thresholds:
                    pred = (change_prob > thresh).long()
                    p_np = pred.cpu().numpy().squeeze().astype(np.uint8)
                    gt_np = gt.numpy().squeeze().astype(np.uint8)
                    m = compute_metrics_np(p_np, gt_np)
                    metrics.append(m)
                    if m[3] > best_f1:
                        best_f1 = m[3]
                        best_threshold = thresh
                
                pred = (change_prob > best_threshold).long()
                p_np = pred.cpu().numpy().squeeze().astype(np.uint8)
                
                p_np = binary_opening(p_np, structure=np.ones((3,3))).astype(np.uint8)
                p_np = binary_closing(p_np, structure=np.ones((3,3))).astype(np.uint8)
                
                if args.use_guided_filter:
                    probs_np = torch.softmax(out, dim=1).cpu().numpy().squeeze()
                    p_np = apply_guided_filter(probs_np, im1[0], radius=5, eps=0.1)
                
                gt_np = gt.numpy().squeeze().astype(np.uint8)
                m = compute_metrics_np(p_np, gt_np)
                city = val_cities[idx]
                print(f"{city}: P={m[0]:.3f}, R={m[1]:.3f}, IoU={m[2]:.3f}, F1={m[3]:.3f}, OA={m[4]:.3f}, Threshold={best_threshold:.2f}")
                
                if m[3] > best_f1:
                    print(f"Updated best: city={city}, F1={m[3]:.3f}")
                    best_f1 = m[3]
                    best_city = city
                    best_pred = p_np
                    best_gt = gt_np
                                
                all_metrics.append(m)
        
        avg = np.mean(all_metrics, axis=0)
        print(f"\nValidation Average → P={avg[0]:.3f}, R={avg[1]:.3f}, IoU={avg[2]:.3f}, F1={avg[3]:.3f}, OA={avg[4]:.3f}")
        print(f"\nBest performing city: {best_city} (F1={best_f1:.3f})")
        
        from PIL import Image
        os.makedirs('outputs', exist_ok=True)
        Image.fromarray((best_pred*255)).save(f"outputs/{best_city}_pred.png")
        Image.fromarray((best_gt*255)).save(f"outputs/{best_city}_gt.png")
        print(f"\nSaved best prediction and ground truth for {best_city} in outputs/")

    else:
        ds = ChangeDetectionDataset(args.root, TEST_CITIES, augment=False, 
                                  require_mask=False, return_time=True, use_all_bands=args.use_all_bands)
        print(f"[INFO] Predicting on {len(ds)} test samples")
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate)

        for i, net in enumerate(nets):
            checkpoint_path = f'checkpoints/fresunet_best_{i}.pth'
            if os.path.exists(checkpoint_path):
                print(f"[INFO] Loading trained model {i+1} from {checkpoint_path}")
                net.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        
        print("\n[INFO] Generating predictions for test cities...")
        for idx, (im1, im2, _, time_diff) in enumerate(loader):
            im1, im2, time_diff = im1.to(device), im2.to(device), time_diff.to(device)
            ensemble_probs = []
            
            with torch.no_grad():
                for net in nets:
                    out, _ = net(im1, im2, time_diff)
                    probs = torch.softmax(out, dim=1)
                    ensemble_probs.append(probs[:, 1, :, :])
                change_prob = torch.mean(torch.stack(ensemble_probs), dim=0)
                pred = (change_prob > 0.7).long()
                
                p_np = pred.cpu().numpy().squeeze().astype(np.uint8)
                p_np = binary_opening(p_np, structure=np.ones((3,3))).astype(np.uint8)
                p_np = binary_closing(p_np, structure=np.ones((3,3))).astype(np.uint8)
                
                if args.use_guided_filter:
                    probs_np = torch.softmax(out, dim=1).cpu().numpy().squeeze()
                    p_np = apply_guided_filter(probs_np, im1[0], radius=5, eps=0.1)
                
                city_name = TEST_CITIES[idx]
                from PIL import Image
                os.makedirs('outputs', exist_ok=True)
                pred_path = f"outputs/{city_name}_pred.png"
                Image.fromarray((p_np*255)).save(pred_path)
                print(f"Saved prediction for {city_name} to {pred_path}")

        print("\n[INFO] Test predictions saved in outputs/")
        print("[NOTE] Ground truth masks not available for test cities - metrics cannot be calculated")