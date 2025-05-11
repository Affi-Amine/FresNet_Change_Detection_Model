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
            out = net(im1, im2)              # [B,2,H,W] raw logits
            pred = torch.argmax(out, dim=1)   # pick the highest‐score class
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
    parser.add_argument('--root',       required=True, help='Path to data/')
    parser.add_argument('--train',      action='store_true', help='Train the model')
    parser.add_argument('--eval-train', action='store_true', help='Evaluate on TRAIN_CITIES')
    parser.add_argument('--eval-val',   action='store_true', help='Evaluate on validation cities')
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patience',   type=int, default=7)
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
    net = FresUNet(input_nbr=6, label_nbr=2).to(device)

    # -- load pre-trained if exists (partial load allowed) --
    pretrained = 'fresunet3_final.pth.tar'
    if os.path.exists(pretrained):
        print(f"[INFO] Loading pre-trained model from {pretrained}")
        ck = torch.load(pretrained, map_location=device)
        state = net.state_dict()
        pd   = ck.get('state_dict', ck)
        # filter only matching shapes
        matched = {k:v for k,v in pd.items() if k in state and v.shape==state[k].shape}
        state.update(matched)
        net.load_state_dict(state)
        print(f"[INFO] Loaded {len(matched)}/{len(state)} params")
    else:
        print(f"[WARNING] No pre-trained weights, training from scratch")

    if args.train:
        train_ds = ChangeDetectionDataset(args.root, TRAIN_CITIES[:10], augment=True,  require_mask=True)
        val_ds   = ChangeDetectionDataset(args.root, TRAIN_CITIES[10:], augment=False, require_mask=True)
        print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=custom_collate)
        val_loader   = DataLoader(val_ds,   batch_size=1,            shuffle=False,
                                  num_workers=2, collate_fn=custom_collate)

        # 2-class CrossEntropy with heavy positive weight
        pw      = train_ds.get_pos_weight() * 6.0
        weights = torch.tensor([1.0, pw], dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                               factor=0.5, patience=3, min_lr=1e-6)
        stopper   = EarlyStopping(patience=args.patience)
        best_f1   = 0.0

        for ep in range(1, args.epochs+1):
            net.train()
            total_loss = 0.0
            for im1, im2, m in train_loader:
                im1, im2, m = im1.to(device), im2.to(device), m.to(device)
                optimizer.zero_grad()
                out  = net(im1, im2)        # [B,2,H,W]
                loss = criterion(out, m.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {ep}/{args.epochs} Loss: {avg_loss:.4f}")

            val_metrics = evaluate_model(net, val_loader, device, desc=f"Validation (ep{ep})")
            f1 = val_metrics[3]
            scheduler.step(f1)

            if f1 > best_f1:
                best_f1 = f1
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(net.state_dict(), 'checkpoints/fresunet_best.pth')
                print(f"[INFO] New best model (F1={best_f1:.3f})")

            stopper(1 - f1)
            if stopper.early_stop:
                print("[INFO] Early stopping — training complete")
                break

    elif args.eval_train:
        # Load best model checkpoint
        cp = 'checkpoints/fresunet_best.pth'
        net.load_state_dict(torch.load(cp, map_location=device))
        net.eval()

        # Prepare dataset and loader
        ds     = ChangeDetectionDataset(args.root, TRAIN_CITIES,        augment=False, require_mask=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate)

        # Evaluate per city and track best F1
        best_f1   = -1.0
        best_city = None
        best_pred = None
        best_gt   = None
        all_metrics = []

        print(f"[INFO] Train Eval on {len(loader)} samples")
        for idx, (im1, im2, gt) in enumerate(loader):
            im1, im2 = im1.to(device), im2.to(device)
            with torch.no_grad():
                out  = net(im1, im2)
                pred = out.argmax(1).cpu().numpy().squeeze().astype(np.uint8)
            gt_np = gt.numpy().squeeze().astype(np.uint8)
            city = TRAIN_CITIES[idx]

            m = compute_metrics_np(pred, gt_np)
            print(f"{city}: P={m[0]:.3f}, R={m[1]:.3f}, IoU={m[2]:.3f}, F1={m[3]:.3f}, OA={m[4]:.3f}")
            all_metrics.append(m)

            if m[3] > best_f1:
                best_f1   = m[3]
                best_city = city
                best_pred = pred.copy()
                best_gt   = gt_np.copy()

        avg = np.mean(all_metrics, axis=0)
        print(f"\nTrain Eval Average → P={avg[0]:.3f}, R={avg[1]:.3f}, IoU={avg[2]:.3f}, F1={avg[3]:.3f}, OA={avg[4]:.3f}")
        print(f"Best performing city (train): {best_city} (F1={best_f1:.3f})")

        from PIL import Image
        os.makedirs('outputs', exist_ok=True)
        Image.fromarray(best_pred * 255).save(f"outputs/{best_city}_train_pred.png")
        Image.fromarray(best_gt   * 255).save(f"outputs/{best_city}_train_gt.png")
        print(f"Saved train best pred/gt for {best_city} in outputs/")

    elif args.eval_val:
        # Load best model checkpoint
        cp = 'checkpoints/fresunet_best.pth'
        net.load_state_dict(torch.load(cp, map_location=device))
        net.eval()

        # Prepare dataset and loader
        val_cities = TRAIN_CITIES[10:]
        ds     = ChangeDetectionDataset(args.root, val_cities, augment=False, require_mask=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate)

        # Evaluate per city and track best F1
        best_f1   = -1.0
        best_city = None
        best_pred = None
        best_gt   = None
        all_metrics = []

        print(f"[INFO] Val Eval on {len(loader)} samples")
        for idx, (im1, im2, gt) in enumerate(loader):
            im1, im2 = im1.to(device), im2.to(device)
            with torch.no_grad():
                out  = net(im1, im2)
                pred = out.argmax(1).cpu().numpy().squeeze().astype(np.uint8)
            gt_np = gt.numpy().squeeze().astype(np.uint8)
            city = val_cities[idx]

            m = compute_metrics_np(pred, gt_np)
            print(f"{city}: P={m[0]:.3f}, R={m[1]:.3f}, IoU={m[2]:.3f}, F1={m[3]:.3f}, OA={m[4]:.3f}")
            all_metrics.append(m)

            if m[3] > best_f1:
                best_f1   = m[3]
                best_city = city
                best_pred = pred.copy()
                best_gt   = gt_np.copy()

        avg = np.mean(all_metrics, axis=0)
        print(f"\nVal Eval Average → P={avg[0]:.3f}, R={avg[1]:.3f}, IoU={avg[2]:.3f}, F1={avg[3]:.3f}, OA={avg[4]:.3f}")
        print(f"Best performing city (val): {best_city} (F1={best_f1:.3f})")

        from PIL import Image
        os.makedirs('outputs', exist_ok=True)
        Image.fromarray(best_pred * 255).save(f"outputs/{best_city}_val_pred.png")
        Image.fromarray(best_gt   * 255).save(f"outputs/{best_city}_val_gt.png")
        print(f"Saved val best pred/gt for {best_city} in outputs/")

    else:
        # Inference on test set
        ds     = ChangeDetectionDataset(args.root, TEST_CITIES, augment=False, require_mask=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate)
        net.load_state_dict(torch.load('checkpoints/fresunet_best.pth', map_location=device))
        net.eval()

        os.makedirs('outputs', exist_ok=True)
        print(f"[INFO] Predicting on {len(ds)} test cities…")
        for idx, (im1, im2, _) in enumerate(loader):
            im1, im2 = im1.to(device), im2.to(device)
            with torch.no_grad():
                out  = net(im1, im2)
                pred = out.argmax(1).cpu().numpy().squeeze().astype(np.uint8)
            city = TEST_CITIES[idx]
            from PIL import Image
            Image.fromarray(pred * 255).save(f"outputs/{city}_pred.png")
            print(f"→ {city}")

        print("[INFO] All test predictions saved in outputs/")
