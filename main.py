
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from fresunet import AttentionFresUNet
from dataset import ChangeDetectionDataset
from scipy.ndimage import binary_opening, binary_closing
import torch.nn as nn
import torch.nn.functional as F
import cv2


# City lists
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
    """Compute Precision, Recall, IoU, F1, OA metrics"""
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
    """Early stopping to prevent overfitting. Stops if score doesn't improve."""
    def __init__(self, patience=7, min_delta=0.0, mode='max'): # mode can be 'min' for loss or 'max' for F1/accuracy
        self.patience = patience
        self.min_delta = min_delta  # Minimum change to qualify as an improvement
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        if self.mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        
        if self.mode == 'max':
            self.best_score = -float('inf') # Initialize best_score appropriately for max mode
        else: # mode == 'min'
            self.best_score = float('inf')  # Initialize best_score appropriately for min mode

    def __call__(self, current_score):
        if self.mode == 'max':
            if current_score > self.best_score + self.min_delta: # Score improved
                self.best_score = current_score
                self.counter = 0
            else: # Score did not improve enough
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        elif self.mode == 'min': # For loss
            if current_score < self.best_score - self.min_delta: # Loss decreased
                self.best_score = current_score
                self.counter = 0
            else: # Loss did not decrease enough
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

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


def plot_training_progress(train_losses, val_f1_scores, save_path=None):
    """Plot training loss and validation F1 scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training loss
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Plot validation F1 score
    ax2.plot(val_f1_scores, 'r-', label='Validation F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1 Score')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
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


def inference_with_test_time_augmentation(net, img1, img2, device, tta_flips=True, tta_scales=True, tta_rotations=True):
    """Perform inference with test-time augmentation
    
    Args:
        net: Network model
        img1, img2: Input image tensors
        device: Computation device
        tta_flips: Apply horizontal and vertical flips
        tta_scales: Apply scale augmentation
        tta_rotations: Apply rotations
        
    Returns:
        Average prediction probability
    """
    net.eval()
    
    # Original prediction
    with torch.no_grad():
        out = net(img1, img2)
        prob = torch.softmax(out, dim=1)[:, 1, :, :]  # Change probability
    
    all_probs = [prob]
    
    # Test-time augmentation with flips
    if tta_flips:
        # Horizontal flip
        img1_h = torch.flip(img1, dims=[3])
        img2_h = torch.flip(img2, dims=[3])
        with torch.no_grad():
            out_h = net(img1_h, img2_h)
            prob_h = torch.softmax(out_h, dim=1)[:, 1, :, :]
        all_probs.append(torch.flip(prob_h, dims=[2]))
        
        # Vertical flip
        img1_v = torch.flip(img1, dims=[2])
        img2_v = torch.flip(img2, dims=[2])
        with torch.no_grad():
            out_v = net(img1_v, img2_v)
            prob_v = torch.softmax(out_v, dim=1)[:, 1, :, :]
        all_probs.append(torch.flip(prob_v, dims=[1]))
        
        # Both flips
        img1_hv = torch.flip(img1, dims=[2, 3])
        img2_hv = torch.flip(img2, dims=[2, 3])
        with torch.no_grad():
            out_hv = net(img1_hv, img2_hv)
            prob_hv = torch.softmax(out_hv, dim=1)[:, 1, :, :]
        all_probs.append(torch.flip(prob_hv, dims=[1, 2]))
    
    # Test-time augmentation with scales
    if tta_scales:
        # Scale down by 10%
        scale_factor = 0.9
        img1_s = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        img2_s = F.interpolate(img2, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        with torch.no_grad():
            out_s = net(img1_s, img2_s)
            prob_s = torch.softmax(out_s, dim=1)[:, 1, :, :]
        
        # Resize back to original
        prob_s = F.interpolate(prob_s.unsqueeze(1), size=prob.shape[1:], mode='bilinear', align_corners=False).squeeze(1)
        all_probs.append(prob_s)
        
        # Scale up by 10%
        scale_factor = 1.1
        img1_s = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        img2_s = F.interpolate(img2, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        # Crop to original size
        _, _, h, w = img1.shape
        img1_s = img1_s[:, :, :h, :w]
        img2_s = img2_s[:, :, :h, :w]
        with torch.no_grad():
            out_s = net(img1_s, img2_s)
            prob_s = torch.softmax(out_s, dim=1)[:, 1, :, :]
        all_probs.append(prob_s)
    
    # Test-time augmentation with rotations
    if tta_rotations:
        # 90 degree rotation
        img1_r = torch.rot90(img1, k=1, dims=[2, 3])
        img2_r = torch.rot90(img2, k=1, dims=[2, 3])
        with torch.no_grad():
            out_r = net(img1_r, img2_r)
            prob_r = torch.softmax(out_r, dim=1)[:, 1, :, :]
        all_probs.append(torch.rot90(prob_r, k=3, dims=[1, 2]))  # Rotate back
    
    # Average all predictions
    avg_prob = torch.stack(all_probs).mean(dim=0)
    return avg_prob

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Change Detection with FresUNet')
    
    # Data parameters
    parser.add_argument('--root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    
    # Model parameters
    parser.add_argument('--pretrained', type=str, default='checkpoints/fresune3_old_best.pth', 
                        help='Path to pretrained model')
    
    # Training parameters
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    
    # Loss function parameters
    parser.add_argument('--focal-weight', type=float, default=0.4, help='Weight for focal loss')
    parser.add_argument('--dice-weight', type=float, default=0.3, help='Weight for dice loss')
    parser.add_argument('--boundary-weight', type=float, default=0.3, help='Weight for boundary loss')
    parser.add_argument('--pos-weight-multiplier', type=float, default=10.0, help='Multiplier for positive class weight')
    
    # Evaluation parameters
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--test', action='store_true', help='Test on test set')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 
                         'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Create model
    print("[INFO] Initializing FresUNet model")
    
    # Set batch norm momentum to a lower value for stability with small batches
    def adjust_bn_momentum(module, momentum=0.1):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = momentum
    
    net = AttentionFresUNet(input_nbr=6, label_nbr=2).to(device)
    net.apply(lambda m: adjust_bn_momentum(m, momentum=0.1))
    
    # Load pre-trained model if available
    if os.path.exists(args.pretrained):
        print(f"[INFO] Loading pre-trained model from {args.pretrained}")
        net = load_model_with_mismatch(net, args.pretrained, device)
    
    # Create datasets
    if args.train or args.eval:
        # Use 70% of train cities for training, 30% for validation
        train_cities = TRAIN_CITIES[:10]  # Use first 10 cities for training
        val_cities = TRAIN_CITIES[10:]    # Use remaining cities for validation
        
        print(f"[INFO] Creating datasets with {len(train_cities)} train cities and {len(val_cities)} val cities")
        # Remove the strong_augment parameter
        train_ds = ChangeDetectionDataset(args.root, train_cities, augment=True, require_mask=True)
        val_ds = ChangeDetectionDataset(args.root, val_cities, augment=False, require_mask=True)
        
        print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Required for MPS compatibility
            pin_memory=False  # Disable pin_memory for MPS
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Required for MPS compatibility
            pin_memory=False  # Disable pin_memory for MPS
        )
    
    # Train the model
    if args.train:
        net, val_metrics = train_model(net, train_loader, val_loader, device, args)
    
    # Evaluate on validation set
    if args.eval and not args.train:
        print("[INFO] Evaluating model on validation set")
        val_metrics, _ = evaluate_model(
            net, val_loader, device, 
            desc="Validation",
            save_path=f"{args.output}/validation"
        )
    
    # Test on test set
    if args.test:
        print("[INFO] Testing model on test set")
        test_ds = ChangeDetectionDataset(args.root, TEST_CITIES, augment=False, require_mask=True)
        test_loader = test_ds.get_loader(batch_size=1)
        
        test_metrics, _ = evaluate_model(
            net, test_loader, device, 
            desc="Test",
            save_path=f"{args.output}/test"
        )

if __name__ == '__main__':
    main()