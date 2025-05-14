# main.py - Enhanced for small change detection

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
import matplotlib.pyplot as plt
from PIL import Image

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
    """Custom collate function to handle tensor batching"""
    img1s = torch.stack([item[0] for item in batch])
    img2s = torch.stack([item[1] for item in batch])
    masks = torch.stack([item[2] for item in batch])
    return img1s, img2s, masks

def compute_metrics_np(pred, gt):
    """Compute Precision, Recall, IoU, F1, OA metrics"""
    tn, fp, fn, tp = confusion_matrix(gt.flatten(), pred.flatten(), labels=[0,1]).ravel()
    p  = tp/(tp+fp) if tp+fp>0 else 0
    r  = tp/(tp+fn) if tp+fn>0 else 0
    i  = tp/(tp+fp+fn) if tp+fp+fn>0 else 0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn>0 else 0
    oa = (tp+tn)/(tp+tn+fp+fn)
    return p, r, i, f1, oa

class FocalLoss(nn.Module):
    """Focal Loss to focus more on hard-to-detect small changes"""
    def __init__(self, gamma=2.0, weight=None, reduction='mean', epsilon=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.epsilon = epsilon
        
    def forward(self, inputs, targets):
        # Apply log_softmax for numerical stability instead of softmax+log
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Get target probabilities
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2) # B, C, H, W
        
        # Calculate focal loss directly with log probabilities
        probs = torch.exp(log_probs)
        # Clamp probabilities to avoid log(0) or log(1) issues indirectly
        probs = torch.clamp(probs, self.epsilon, 1.0 - self.epsilon)
        
        focal_weight_term = (1 - probs) ** self.gamma
        
        # Apply class weights if provided
        if self.weight is not None:
            # Reshape weight to match targets (B, C, H, W) or (C)
            weight_ = self.weight.view(1, -1, 1, 1) if self.weight.ndim == 1 else self.weight
            focal_weight_term = focal_weight_term * weight_
        
        # Calculate loss with stability safeguards
        # Cross-entropy term: -targets_one_hot * log_probs
        # Sum over classes dimension
        focal_loss = -focal_weight_term * targets_one_hot * log_probs
        focal_loss = torch.sum(focal_loss, dim=1)  # Sum over classes, result shape (B, H, W)
        
        # Check for NaN values and replace with zeros
        if torch.isnan(focal_loss).any():
            print("[WARNING] NaN values detected in focal loss, replacing with zeros")
            focal_loss = torch.where(torch.isnan(focal_loss), torch.zeros_like(focal_loss), focal_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
def load_model_with_mismatch(model, checkpoint_path, device):
    """Load pre-trained weights while skipping layers with size mismatch"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model_dict = model.state_dict()
    
    # Filter out mismatched layers
    compatible_dict = {}
    incompatible = []
    
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                compatible_dict[k] = v
            else:
                incompatible.append(f"{k}: checkpoint={v.shape}, model={model_dict[k].shape}")
        else:
            # Checkpoint has keys not in model
            pass
    
    # Find keys in model that are not in checkpoint
    missing_keys = [k for k in model_dict.keys() if k not in state_dict]
    
    # Print loading statistics
    total_params = len(model_dict)
    loaded_params = len(compatible_dict)
    print(f"[INFO] Loaded {loaded_params}/{total_params} parameters")
    print(f"[INFO] {len(missing_keys)} parameters in model not found in checkpoint")
    print(f"[INFO] {len(incompatible)} parameters had shape mismatches")
    
    # Load compatible parameters
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict, strict=False)
    
    return model

class DiceLoss(nn.Module):
    """Dice Loss for better segmentation of small objects"""
    def __init__(self, smooth=1.0, epsilon=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.epsilon = epsilon
        
    def forward(self, inputs, targets):
        # Get probabilities for prediction with clamping for stability
        inputs_probs = F.softmax(inputs, dim=1)
        inputs_probs = torch.clamp(inputs_probs, self.epsilon, 1.0 - self.epsilon)
        
        # Only consider the "change" class (index 1)
        inputs_change_prob = inputs_probs[:, 1, :, :]
        
        # Flatten
        inputs_flat = inputs_change_prob.reshape(-1)
        targets_flat = targets.reshape(-1).float()
        
        # Calculate Dice with numerical stability
        intersection = (inputs_flat * targets_flat).sum()
        dice_numerator = 2. * intersection + self.smooth
        dice_denominator = inputs_flat.sum() + targets_flat.sum() + self.smooth
        
        # Safeguard against division by zero or very small denominator
        if dice_denominator.item() < self.epsilon:
            print("[WARNING] Near-zero denominator in Dice loss, returning max loss (1.0)")
            return torch.tensor(1.0, device=inputs.device, dtype=torch.float32)
        
        dice = dice_numerator / dice_denominator
        loss = 1 - dice
        
        # Check for NaN and replace with fallback
        if torch.isnan(loss).any():
            print("[WARNING] NaN values detected in Dice loss, returning max loss (1.0)")
            return torch.tensor(1.0, device=inputs.device, dtype=torch.float32)
        
        return loss

class BoundaryLoss(nn.Module):
    """Boundary Loss to focus on the edges of changed regions"""
    def __init__(self, weight=1.0, epsilon=1e-7):
        super(BoundaryLoss, self).__init__()
        self.weight = weight
        self.epsilon = epsilon
        
        # Define Sobel kernels for edge detection
        self.sobel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        
    def forward(self, inputs, targets):
        device = inputs.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        
        # Extract "change" class probabilities with clamping for stability
        probs = F.softmax(inputs, dim=1)[:, 1:2, :, :]  # Keep dim for conv (B, 1, H, W)
        probs = torch.clamp(probs, self.epsilon, 1.0 - self.epsilon)
        
        # Compute edges for predictions
        pred_edges_x = F.conv2d(probs, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(probs, self.sobel_y, padding=1)
        # Add epsilon inside sqrt for stability
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + self.epsilon)
        # Add gradient scaling
        pred_edges = pred_edges * 2.0  # Amplify edge signals
        
        # Compute edges for targets (convert to float first)
        targets_float = targets.float().unsqueeze(1)  # Add channel dim (B, 1, H, W)
        target_edges_x = F.conv2d(targets_float, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(targets_float, self.sobel_y, padding=1)
        # Add epsilon inside sqrt for stability
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + self.epsilon)
        
        # Mean squared error on edges
        loss = F.mse_loss(pred_edges, target_edges)

        # Check for NaN and replace with zero (or a large value if preferred)
        if torch.isnan(loss).any():
            print("[WARNING] NaN values detected in Boundary loss, replacing with zero")
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        
        return self.weight * loss

class CombinedLoss(nn.Module):
    """Combined loss function with weights for different loss components"""
    def __init__(self, weight=None, focal_weight=0.5, dice_weight=0.3, boundary_weight=0.2, gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.weight = weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.gamma = gamma  # Store gamma parameter
        
        self.focal_loss = FocalLoss(weight=weight, gamma=gamma)  # Pass gamma to FocalLoss
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        
    def forward(self, inputs, targets):
        # Calculate all loss components with NaN detection
        try:
            fl = self.focal_loss(inputs, targets)
            if torch.isnan(fl).any() or torch.isinf(fl).any():
                print("[WARNING] NaN or Inf detected in focal loss component")
                fl = torch.tensor(0.1, device=inputs.device)
                
            dl = self.dice_loss(inputs, targets)
            if torch.isnan(dl).any() or torch.isinf(dl).any():
                print("[WARNING] NaN or Inf detected in dice loss component")
                dl = torch.tensor(0.1, device=inputs.device)
                
            bl = self.boundary_loss(inputs, targets)
            if torch.isnan(bl).any() or torch.isinf(bl).any():
                print("[WARNING] NaN or Inf detected in boundary loss component")
                bl = torch.tensor(0.1, device=inputs.device)
            
            # Combine with weights
            total_loss = (self.focal_weight * fl +
                         self.dice_weight * dl +
                         self.boundary_weight * bl)
            
            # Final NaN check
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                print("[WARNING] NaN or Inf detected in combined loss, using fallback value")
                total_loss = torch.tensor(0.1, device=inputs.device)
                
            return total_loss
            
        except Exception as e:
            print(f"[ERROR] Exception in loss calculation: {e}")
            # Return a small fallback loss to continue training
            return torch.tensor(0.1, device=inputs.device)

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

def evaluate_model(net, loader, device, desc="Eval", save_path=None, threshold=0.4, epoch=0, current_val_f1=0.0, apply_temp_scaling=False):
    """Evaluate model and optionally save results"""
    net.eval()
    all_metrics = []
    city_metrics = {}
    all_preds = []
    all_gts = []
    
    print(f"[INFO] {desc} on {len(loader)} samples")
    
    # Use higher threshold for better precision
    print(f"[DEBUG] Using threshold: {threshold:.2f}")
    total_positive_pixels = 0
    total_pixels = 0
    
    for idx, (im1, im2, gt) in enumerate(loader):
        im1, im2 = im1.to(device), im2.to(device)
        city = loader.dataset.samples[idx][0].split('/')[-3]  # Extract city name
        
        with torch.no_grad():
            # Forward pass
            out = net(im1, im2)
            
            # Remove temperature scaling - no longer dividing by 1.5
            
            # Get change probability
            probs = torch.softmax(out, dim=1)
            change_prob = probs[:, 1]
            
            # Use a higher threshold to improve precision
            pred = (change_prob > threshold).long()
            
            # Apply enhanced morphological cleaning
            if pred.sum() > 0:  # Only if we have positive predictions
                # Convert to numpy for morphological operations
                pred_np = pred.cpu().numpy().squeeze()
                
                # Import morphology functions
                from skimage import morphology
                
                # Step 1: Remove very small isolated positive predictions (false positives)
                # Increased min_size from 5 to 8 for better noise removal
                cleaned = morphology.remove_small_objects(pred_np.astype(bool), min_size=8)
                
                # Step 2: Fill small holes in positive regions (false negatives in change areas)
                # Increased area_threshold from 10 to 15 for better hole filling
                cleaned = morphology.remove_small_holes(cleaned, area_threshold=15)
                
                # Step 3: Apply a small closing operation to connect nearby positive regions
                cleaned = morphology.binary_closing(cleaned, morphology.disk(2))
                
                # Step 4: Apply a small opening to remove thin connections
                cleaned = morphology.binary_opening(cleaned, morphology.disk(1))
                
                # Step 5: Remove small objects again after morphological operations
                cleaned = morphology.remove_small_objects(cleaned, min_size=10)
                
                # Convert back to tensor
                pred = torch.from_numpy(cleaned.astype(np.int64)).to(device)
        
        p_np = pred.cpu().numpy().squeeze().astype(np.uint8)
        gt_np = gt.numpy().squeeze().astype(np.uint8)
        
        # Track statistics for debugging
        total_positive_pixels += np.sum(gt_np)
        total_pixels += gt_np.size
        pred_positive = np.sum(p_np)
        
        # Print prediction statistics
        print(f"[DEBUG] {city}: GT positive pixels: {np.sum(gt_np)}/{gt_np.size} ({np.sum(gt_np)/gt_np.size*100:.2f}%)")
        print(f"[DEBUG] {city}: Pred positive pixels: {pred_positive}/{p_np.size} ({pred_positive/p_np.size*100:.2f}%)")
        print(f"[DEBUG] {city}: Max probability: {change_prob.max().item():.4f}")
        
        # Calculate metrics
        tp = np.sum((p_np == 1) & (gt_np == 1))
        fp = np.sum((p_np == 1) & (gt_np == 0))
        fn = np.sum((p_np == 0) & (gt_np == 1))
        tn = np.sum((p_np == 0) & (gt_np == 0))
        
        # Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        overall_acc = (tp + tn) / (tp + tn + fp + fn)
        
        metrics = [precision, recall, iou, f1, overall_acc]
        all_metrics.append(metrics)
        city_metrics[city] = metrics
        
        print(f"{city}: P={precision:.3f}, R={recall:.3f}, IoU={iou:.3f}, F1={f1:.3f}, OA={overall_acc:.3f}")
        
        # Store predictions and ground truth for visualization
        all_preds.append(p_np)
        all_gts.append(gt_np)
    
    # Calculate average metrics
    avg = np.mean(all_metrics, axis=0)
    print(f"{desc} Average → P={avg[0]:.3f}, R={avg[1]:.3f}, IoU={avg[2]:.3f}, F1={avg[3]:.3f}, OA={avg[4]:.3f}")
    
    # Print overall class distribution statistics
    print(f"[DEBUG] Overall: GT positive pixels: {total_positive_pixels}/{total_pixels} ({total_positive_pixels/total_pixels*100:.2f}%)")
    print(f"[DEBUG] Class imbalance ratio: {(total_pixels-total_positive_pixels)/total_positive_pixels:.2f}:1 (negative:positive)")
    print(f"[DEBUG] Recommended pos_weight_multiplier: {min(25.0, (total_pixels-total_positive_pixels)/total_positive_pixels):.2f}")
    
    # Save results if path is given
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Find best and worst cities
        best_city = max(city_metrics.items(), key=lambda x: x[1][3])[0]  # By F1 score
        worst_city = min(city_metrics.items(), key=lambda x: x[1][3])[0]
        
        print(f"Best city: {best_city}, Worst city: {worst_city}")
        
        # Save predictions and ground truth for these cities
        for city_name, (pred, gt) in zip([best_city, worst_city], [(all_preds[i], all_gts[i]) for i in range(len(all_preds))]):
            Image.fromarray((pred*255)).save(f"{save_path}/{city_name}_pred.png")
            Image.fromarray((gt*255)).save(f"{save_path}/{city_name}_gt.png")
            
            # Create visualization with overlays
            visualize_prediction(pred, gt, save_path=f"{save_path}/{city_name}_overlay.png")
    
    return avg, city_metrics

def visualize_prediction(prediction, ground_truth, save_path=None):
    """Create a visualization that shows prediction vs ground truth"""
    plt.figure(figsize=(12, 6))
    
    # Plot ground truth
    plt.subplot(1, 3, 1)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Plot prediction
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    # Create overlay: Green = True Positive, Red = False Negative, Blue = False Positive
    overlay = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = ((prediction == 1) & (ground_truth == 1)) * 255  # True Positives (Green)
    overlay[..., 0] = ((prediction == 0) & (ground_truth == 1)) * 255  # False Negatives (Red)
    overlay[..., 2] = ((prediction == 1) & (ground_truth == 0)) * 255  # False Positives (Blue)
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay (TP=Green, FN=Red, FP=Blue)')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def train_model(net, train_loader, val_loader, device, args):
    """Train the model with focus on small change detection"""
    # Create directories for output
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Early stopping with increased patience for better convergence
    early_stopping = EarlyStopping(patience=args.patience + 5, min_delta=0.001, mode='max') 
    
    # Training loop
    best_val_f1 = 0
    train_losses = []  
    val_f1_scores = []
    
    # Initialize optimizer with stronger weight decay to prevent overfitting
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay * 2.0)  # Double weight decay
    
    # Get class weights for loss function - REDUCE weight for positive class to avoid predicting everything as positive
    base_pos_weight = train_loader.dataset.get_pos_weight()
    # Use a moderate multiplier reduction (50% instead of 80%)
    effective_pos_weight = base_pos_weight * args.pos_weight_multiplier * 0.5  # Reduce by 50%
    
    # Cap the final positive weight to prevent extreme values
    max_allowed_pos_weight = 2.0  
    final_pos_weight = min(effective_pos_weight, max_allowed_pos_weight)
    
    weights = torch.tensor([1.0, final_pos_weight], dtype=torch.float32, device=device)
    
    # Print detailed class weight information
    print(f"\n[DEBUG] Dataset base positive class weight (from get_pos_weight()): {base_pos_weight:.2f}")
    print(f"[DEBUG] Args positive class weight multiplier: {args.pos_weight_multiplier:.2f}")
    print(f"[DEBUG] Effective positive class weight (base * multiplier * 0.5): {effective_pos_weight:.2f}")
    print(f"[DEBUG] Final positive class weight (after cap at {max_allowed_pos_weight:.2f}): {final_pos_weight:.2f}")
    print(f"[DEBUG] Final class weights tensor for loss: {weights}")
    
    # Adjust loss component weights to focus more on precision
    criterion = CombinedLoss(
        weight=weights,
        focal_weight=0.4,    # Increased to focus on hard examples
        dice_weight=0.3,     # Maintained for overall shape consistency
        boundary_weight=0.3, # Reduced slightly to balance
        gamma=4.0            # Increased gamma for more focus on hard examples
    )
    
    # Improved learning rate scheduler with more patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Dynamic threshold adjustment - start with a lower threshold
    initial_threshold = 0.3  # Start with a lower threshold to detect more changes
    best_threshold = initial_threshold
    
    # Warm-up phase with gentler learning rate
    print("[INFO] Starting with brief warm-up phase (slightly higher learning rate)")
    warm_up_lr = args.lr * 1.5
    warm_up_iterations = 50
    
    # Set initial learning rate for warm-up
    for param_group in optimizer.param_groups:
        param_group['lr'] = warm_up_lr
    
    # Warm-up loop
    for i in range(warm_up_iterations):
        # Get a batch
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            # If we've gone through the entire dataset, reset
            batch = next(iter(train_loader))
        
        # Unpack batch
        im1, im2, gt = batch
        im1, im2, gt = im1.to(device), im2.to(device), gt.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        out = net(im1, im2)
        
        # No temperature scaling
        loss = criterion(out, gt)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Warm-up iteration {i}/{warm_up_iterations}, Loss: {loss.item():.4f}")
    
    # Reset learning rate to normal value after warm-up
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    print(f"[INFO] Warm-up complete, reverting to normal learning rate: {args.lr}")
    
    # Check predictions after warm-up
    with torch.no_grad():
        net.eval()
        sample_batch = next(iter(train_loader))
        sample_out = net(sample_batch[0].to(device), sample_batch[1].to(device))
        # No temperature scaling
        sample_probs = torch.softmax(sample_out, dim=1)
        print(f"[DEBUG] After warm-up - Change class prob - Max: {sample_probs[:,1].max().item():.4f}, Mean: {sample_probs[:,1].mean().item():.4f}")
        print(f"[DEBUG] Positive predictions: {(sample_probs[:,1] > best_threshold).sum().item()}/{sample_probs[:,1].numel()}")
        net.train()
    
    # Create a function to get balanced batches for curriculum learning
    def get_balanced_batch(loader, difficulty_level):
        """Get a batch with a certain difficulty level (0-1)
        Lower difficulty = more balanced batch (higher % of change pixels)"""
        max_attempts = 5
        for _ in range(max_attempts):
            batch = next(iter(loader))
            im1, im2, gt = batch
            
            # Calculate percentage of positive pixels
            pos_ratio = gt.float().mean().item()
            
            # Early in training (low difficulty), we want more balanced batches (higher pos_ratio)
            # Later in training (high difficulty), we accept more imbalanced batches (lower pos_ratio)
            target_min_ratio = 0.05 * (1 - difficulty_level)  # Starts at 0.05, decreases to 0
            
            if pos_ratio >= target_min_ratio:
                return batch
        
        # If we couldn't find a good batch, return the last one
        return batch
    
    # Implementing curriculum learning
    print("[INFO] Implementing curriculum learning strategy")
    
    # Main training loop
    for ep in range(args.epochs):
        # Calculate curriculum difficulty (0 to 1)
        # Start with easier examples (more balanced) and gradually increase difficulty
        curriculum_difficulty = min(1.0, ep / (args.epochs * 0.5))  # Reaches max difficulty halfway through
        print(f"[INFO] Epoch {ep+1}/{args.epochs} - Curriculum difficulty: {curriculum_difficulty:.2f}")
        
        # Training phase
        net.train()
        running_loss = 0.0
        batch_count = 0
        
        # Use curriculum learning for first half of training
        if curriculum_difficulty < 1.0:
            for batch_idx in range(len(train_loader)):
                # Get batch based on current difficulty
                im1, im2, gt = get_balanced_batch(train_loader, curriculum_difficulty)
                im1, im2, gt = im1.to(device), im2.to(device), gt.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                out = net(im1, im2)
                
                loss = criterion(out, gt)
                
                # Add L1 regularization to encourage sparsity (fewer positive predictions)
                l1_lambda = 0.0001
                l1_norm = sum(p.abs().sum() for p in net.parameters())
                loss = loss + l1_lambda * l1_norm
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                batch_count += 1
                
                # Print progress
                if batch_idx % 5 == 0:
                    print(f"Epoch {ep+1}/{args.epochs} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
        else:
            # Regular training for second half
            for batch_idx, (im1, im2, gt) in enumerate(train_loader):
                im1, im2, gt = im1.to(device), im2.to(device), gt.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                out = net(im1, im2)
                
                loss = criterion(out, gt)
                
                # Add L1 regularization to encourage sparsity (fewer positive predictions)
                l1_lambda = 0.0001
                l1_norm = sum(p.abs().sum() for p in net.parameters())
                loss = loss + l1_lambda * l1_norm
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                batch_count += 1
                
                # Print progress
                if batch_idx % 5 == 0:
                    print(f"Epoch {ep+1}/{args.epochs} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        # Calculate average loss for epoch
        avg_loss = running_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_loss)
        
        # Validation phase
        net.eval()
        # Dynamically adjust threshold based on training progress
        # Start with low threshold and gradually increase it
        current_threshold = min(0.5, initial_threshold + (ep * 0.01))  # Gradually increase threshold
        # Try:
        current_threshold = 0.8  
        
        val_metrics, _ = evaluate_model(net, val_loader, device, epoch=ep+1, 
                                       desc=f"Validation (ep{ep+1})", 
                                       threshold=current_threshold, 
                                       current_val_f1=best_val_f1)
        
        val_f1 = val_metrics[3]  # F1 score is the 4th metric
        val_f1_scores.append(val_f1)
        
        # Learning rate scheduler step
        scheduler.step(val_f1)
        
        # Check for improvement and save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = current_threshold  # Save the best threshold
            torch.save({'state_dict': net.state_dict()}, 'checkpoints/best_model.pth.tar')
            print(f"[INFO] Saved new best model with F1 score: {best_val_f1:.4f} at threshold {best_threshold:.2f}")
            
        # Early stopping check
        early_stopping(val_f1)
        if early_stopping.early_stop:
            print("[INFO] Early stopping triggered")
            break
            
        print(f"Epoch {ep+1}/{args.epochs}  Loss: {avg_loss:.4f}")
    
    # Load best model for final evaluation
    print("[INFO] Loading best model for final evaluation")
    net = load_model_with_mismatch(net, 'checkpoints/best_model.pth.tar', device)
    
    # Final evaluation
    final_metrics, city_metrics = evaluate_model(
        net, val_loader, device, 
        desc="Final Validation", 
        save_path="outputs", 
        threshold=best_threshold
    )
    
    return net, final_metrics

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
        plt.show()

def postprocess_prediction(pred_prob, threshold=0.5, min_area=5):
    """Apply post-processing to improve predictions
    
    Args:
        pred_prob: Prediction probability map (numpy array)
        threshold: Probability threshold
        min_area: Minimum connected component area
        
    Returns:
        Processed binary prediction
    """
    from skimage import morphology
    
    # Apply threshold
    binary = (pred_prob > threshold).astype(np.uint8)
    
    # Remove small components
    labeled = morphology.label(binary)
    component_sizes = np.bincount(labeled.ravel())
    too_small = component_sizes < min_area
    too_small[0] = False  # Keep background
    binary = ~too_small[labeled]
    
    # Optional: Apply morphological operations to clean boundaries
    binary = morphology.binary_opening(binary, morphology.disk(1))
    
    return binary.astype(np.uint8)

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