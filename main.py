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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Path to data/')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval-train', action='store_true', help='Evaluate on TRAIN_CITIES')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
    net = FresUNet(input_nbr=6, label_nbr=2).to(device)

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
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)  
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                             factor=0.5, patience=3, min_lr=1e-6)
        
        # Get class weights for loss function
        pos_weight = train_ds.get_pos_weight() * 0.8  # Reduce positive weight to improve precision
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
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

    else:
        # Inference on TEST_CITIES
        ds = ChangeDetectionDataset(args.root, TEST_CITIES, augment=False, require_mask=False)
        print(f"[INFO] Predicting on {len(ds)} test samples")
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate)

        # Load best weights
        net.load_state_dict(torch.load('checkpoints/fresunet_best.pth', map_location=device))
        net.eval()

        os.makedirs('outputs', exist_ok=True)
        for idx, (im1, im2, _) in enumerate(loader):
            im1, im2 = im1.to(device), im2.to(device)
            with torch.no_grad():
                out = net(im1, im2)
                _, pred = torch.max(out, 1)
            p_np = pred.cpu().numpy().squeeze().astype(np.uint8)
            from PIL import Image
            Image.fromarray((p_np*255)).save(f"outputs/pred_{idx:03d}.png")
        print("[INFO] Test predictions saved in outputs/")
