# main.py

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from fresunet import FresUNet
from dataset import ChangeDetectionDataset

TRAIN_CITIES = [
    'aguasclaras','bercy','bordeaux','nantes','paris','rennes','saclay_e',
    'abudhabi','cupertino','pisa','beihai','hongkong','beirut','mumbai'
]
TEST_CITIES  = [
    'brasilia','montpellier','norcia','rio','saclay_w','valencia',
    'dubai','lasvegas','milano','chongqing'
]

def custom_collate(batch):
    """Custom collate function to handle tensor batching"""
    img1s = torch.stack([item[0] for item in batch])
    img2s = torch.stack([item[1] for item in batch])
    masks = torch.stack([item[2] for item in batch])
    return img1s, img2s, masks

def compute_metrics_np(pred, gt):
    tn, fp, fn, tp = confusion_matrix(gt.flatten(), pred.flatten(), labels=[0,1]).ravel()
    p  = tp/(tp+fp) if tp+fp>0 else 0
    r  = tp/(tp+fn) if tp+fn>0 else 0
    i  = tp/(tp+fp+fn) if tp+fp+fn>0 else 0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn>0 else 0
    oa = (tp+tn)/(tp+tn+fp+fn)
    return p, r, i, f1, oa

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

def evaluate_model(net, loader, device, desc="Eval"):
    net.eval()
    all_metrics = []
    print(f"[INFO] {desc} on {len(loader)} samples")
    
    for idx, (im1, im2, gt) in enumerate(loader):
        im1, im2 = im1.to(device), im2.to(device)
        with torch.no_grad():
            out = net(im1, im2)
            _, pred = torch.max(out, 1)
        p_np = pred.cpu().numpy().squeeze().astype(np.uint8)
        gt_np = gt.numpy().squeeze().astype(np.uint8)
        m = compute_metrics_np(p_np, gt_np)
        city = TRAIN_CITIES[idx % len(TRAIN_CITIES)]
        print(f"{city}: P={m[0]:.3f}, R={m[1]:.3f}, IoU={m[2]:.3f}, F1={m[3]:.3f}, OA={m[4]:.3f}")
        all_metrics.append(m)

    avg = np.mean(all_metrics, axis=0)
    print(f"{desc} Average → P={avg[0]:.3f}, R={avg[1]:.3f}, IoU={avg[2]:.3f}, F1={avg[3]:.3f}, OA={avg[4]:.3f}")
    return avg

def dice_loss(pred, target, smooth=1.0):
    """Calculate Dice loss for segmentation."""
    pred = pred.sigmoid()  # Convert logits to probabilities
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def combined_loss(pred, target, pos_weight, alpha=0.5, beta=0.5):
    """
    Combined BCE and Dice loss with class weights.
    
    Args:
        pred: Model predictions (logits)
        target: Ground truth
        pos_weight: Weight for positive class
        alpha: Weight for BCE loss
        beta: Weight for Dice loss
    """
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_loss = bce(pred, target)
    dice = dice_loss(pred, target)
    return alpha * bce_loss + beta * dice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Path to data/')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval-train', action='store_true', help='Evaluate on TRAIN_CITIES')
    parser.add_argument('--eval-val', action='store_true', help='Evaluate on validation cities')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
    net = FresUNet(input_nbr=6, label_nbr=2).to(device)

    # Load pre-trained model
    pretrained_path = 'fresunet3_final.pth.tar'
    if os.path.exists(pretrained_path):
        print(f"[INFO] Loading pre-trained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
    else:
        print(f"[WARNING] Pre-trained model not found at {pretrained_path}, starting from scratch")

    if args.train:
        # Create datasets
        train_ds = ChangeDetectionDataset(args.root, TRAIN_CITIES[:10], augment=True, require_mask=True)
        val_ds = ChangeDetectionDataset(args.root, TRAIN_CITIES[10:], augment=False, require_mask=True)
        print(f"[INFO] Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                                num_workers=2, collate_fn=custom_collate)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                              num_workers=2, collate_fn=custom_collate)

        # Initialize training components
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)  
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                             factor=0.5, patience=3, min_lr=1e-6)
        
        # Get class weights for loss function
        pos_weight = train_ds.get_pos_weight() * 0.9  # Slightly more balanced
        pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion = lambda pred, target: combined_loss(pred, target, pos_weight, alpha=0.5, beta=0.5)
        early_stopping = EarlyStopping(patience=args.patience)

        best_val_f1 = 0
        for ep in range(args.epochs):
            # Training
            net.train()
            running_loss = 0.0
            for im1, im2, m in train_loader:
                im1, im2, m = im1.to(device), im2.to(device), m.to(device)
                optimizer.zero_grad()
                out = net(im1, im2)
                # Get logits for positive class (change) only
                out = out[:, 1]  # Take channel 1 (positive class)
                m = m.float()  # Convert target to float for BCE loss
                loss = criterion(out, m)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            print(f"Epoch {ep+1}/{args.epochs}  Loss: {train_loss:.4f}")

            # Validation
            val_metrics = evaluate_model(net, val_loader, device, "Validation")
            val_f1 = val_metrics[3]  # F1 score
            scheduler.step(val_f1)  
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(net.state_dict(), 'checkpoints/fresunet_best.pth')
                print(f"[INFO] New best model saved (F1: {val_f1:.3f})")
            
            # Early stopping
            early_stopping(1 - val_f1)
            if early_stopping.early_stop:
                print("[INFO] Early stopping triggered")
                break

        print("[INFO] Training completed")

    elif args.eval_train:
        # Load best weights
        ckpt = 'checkpoints/fresunet_best.pth'
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
        net.load_state_dict(torch.load(ckpt, map_location=device))
        
        # Evaluate on all training cities
        ds = ChangeDetectionDataset(args.root, TRAIN_CITIES, augment=False, require_mask=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate)
        evaluate_model(net, loader, device, "Train Eval")

    elif args.eval_val:
        # Evaluate on validation cities
        val_cities = TRAIN_CITIES[10:]  # Using the validation split
        ds = ChangeDetectionDataset(args.root, val_cities, augment=False, require_mask=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate)
        
        # Load best weights
        checkpoint_path = 'checkpoints/fresunet_best.pth'
        if os.path.exists(checkpoint_path):
            print(f"[INFO] Loading trained model from {checkpoint_path}")
            net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"[WARNING] No trained model found at {checkpoint_path}, using pre-trained model")
        
        net.eval()
        os.makedirs('outputs', exist_ok=True)
        
        print("\n[INFO] Validation Results:")
        all_metrics = []
        best_f1 = -1
        best_city = None
        best_pred = None
        best_gt = None
        
        for idx, (im1, im2, gt) in enumerate(loader):
            im1, im2 = im1.to(device), im2.to(device)
            with torch.no_grad():
                out = net(im1, im2)
                _, pred = torch.max(out, 1)
            
            p_np = pred.cpu().numpy().squeeze().astype(np.uint8)
            gt_np = gt.numpy().squeeze().astype(np.uint8)
            
            # Calculate metrics
            m = compute_metrics_np(p_np, gt_np)
            city = val_cities[idx]
            print(f"{city}: P={m[0]:.3f}, R={m[1]:.3f}, IoU={m[2]:.3f}, F1={m[3]:.3f}, OA={m[4]:.3f}")
            
            # Save if best F1 score
            if m[3] > best_f1:
                best_f1 = m[3]
                best_city = city
                best_pred = p_np
                best_gt = gt_np
            
            all_metrics.append(m)
        
        # Print average and best metrics
        avg = np.mean(all_metrics, axis=0)
        print(f"\nValidation Average → P={avg[0]:.3f}, R={avg[1]:.3f}, IoU={avg[2]:.3f}, F1={avg[3]:.3f}, OA={avg[4]:.3f}")
        print(f"\nBest performing city: {best_city} (F1={best_f1:.3f})")
        
        # Save best prediction and ground truth
        from PIL import Image
        Image.fromarray((best_pred*255)).save(f"outputs/{best_city}_pred.png")
        Image.fromarray((best_gt*255)).save(f"outputs/{best_city}_gt.png")
        print(f"\nSaved best prediction and ground truth for {best_city} in outputs/")

    else:
        # Inference on TEST_CITIES
        ds = ChangeDetectionDataset(args.root, TEST_CITIES, augment=False, require_mask=False)  # Don't require masks
        print(f"[INFO] Predicting on {len(ds)} test samples")
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate)

        # Load best weights from our training
        checkpoint_path = 'checkpoints/fresunet_best.pth'
        if os.path.exists(checkpoint_path):
            print(f"[INFO] Loading trained model from {checkpoint_path}")
            net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"[WARNING] No trained model found at {checkpoint_path}, using pre-trained model")
            
        net.eval()
        os.makedirs('outputs', exist_ok=True)
        
        print("\n[INFO] Generating predictions for test cities...")
        for idx, (im1, im2, _) in enumerate(loader):
            im1, im2 = im1.to(device), im2.to(device)
            with torch.no_grad():
                out = net(im1, im2)
                _, pred = torch.max(out, 1)
            
            p_np = pred.cpu().numpy().squeeze().astype(np.uint8)
            
            # Save prediction
            city_name = TEST_CITIES[idx]
            from PIL import Image
            pred_path = f"outputs/{city_name}_pred.png"
            Image.fromarray((p_np*255)).save(pred_path)
            print(f"Saved prediction for {city_name} to {pred_path}")

        print("\n[INFO] Test predictions saved in outputs/")
        print("[NOTE] Ground truth masks not available for test cities - metrics cannot be calculated")
