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

def compute_metrics_np(pred, gt):
    tn, fp, fn, tp = confusion_matrix(gt.flatten(), pred.flatten(), labels=[0,1]).ravel()
    p  = tp/(tp+fp) if tp+fp>0 else 0
    r  = tp/(tp+fn) if tp+fn>0 else 0
    i  = tp/(tp+fp+fn) if tp+fp+fn>0 else 0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn>0 else 0
    oa = (tp+tn)/(tp+tn+fp+fn)
    return p, r, i, f1, oa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',       required=True, help='Path to data/')
    parser.add_argument('--train',      action='store_true', help='Train the model')
    parser.add_argument('--eval-train', action='store_true', help='Evaluate on TRAIN_CITIES')
    parser.add_argument('--epochs',     type=int, default=50)
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
    net = FresUNet(input_nbr=6, label_nbr=2).to(device)

    if args.train:
        # Training as before
        ds = ChangeDetectionDataset(args.root, TRAIN_CITIES, augment=True, require_mask=True)
        print(f"[INFO] Training samples: {len(ds)}")
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for ep in range(args.epochs):
            net.train()
            running = 0.0
            for im1, im2, m in loader:
                im1, im2, m = im1.to(device), im2.to(device), m.to(device)
                out = net(im1, im2)
                loss = criterion(out, m)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()
            print(f"Epoch {ep+1}/{args.epochs}  Loss: {running/len(loader):.4f}")

        os.makedirs('checkpoints', exist_ok=True)
        torch.save(net.state_dict(), 'checkpoints/fresunet_aug.pth')
        print("[INFO] Model saved to checkpoints/fresunet_aug.pth")

    elif args.eval_train:
        # Load weights
        ckpt = 'checkpoints/fresunet_aug.pth'
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.eval()

        # Dataset with masks required
        ds = ChangeDetectionDataset(args.root, TRAIN_CITIES, augment=False, require_mask=True)
        print(f"[INFO] Evaluating on {len(ds)} train samples")
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

        all_metrics = []
        for idx, (im1, im2, gt) in enumerate(loader):
            im1, im2 = im1.to(device), im2.to(device)
            with torch.no_grad():
                out = net(im1, im2)
                _, pred = torch.max(out, 1)
            p_np = pred.cpu().numpy().squeeze().astype(np.uint8)
            gt_np= gt.numpy().squeeze().astype(np.uint8)
            city = TRAIN_CITIES[idx % len(TRAIN_CITIES)]
            m = compute_metrics_np(p_np, gt_np)
            print(f"{city}: P={m[0]:.3f}, R={m[1]:.3f}, IoU={m[2]:.3f}, F1={m[3]:.3f}, OA={m[4]:.3f}")
            all_metrics.append(m)

        avg = np.mean(all_metrics, axis=0)
        print("Train Eval Average → " +
              f"P={avg[0]:.3f}, R={avg[1]:.3f}, IoU={avg[2]:.3f}, F1={avg[3]:.3f}, OA={avg[4]:.3f}")

    else:
        # Inference on TEST_CITIES only (no metrics)
        ds = ChangeDetectionDataset(args.root, TEST_CITIES, augment=False, require_mask=False)
        print(f"[INFO] Predicting on {len(ds)} test samples")
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

        # Load weights
        net.load_state_dict(torch.load('checkpoints/fresunet_aug.pth', map_location=device))
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
