# focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dealing with class imbalance and hard examples.
    
    The loss adds a factor (1 - pt)^gamma to the standard CE loss,
    where pt is the model's estimated probability for the correct class.
    
    This focuses training on hard/misclassified examples while down-weighting
    easy/well-classified examples.
    
    Parameters:
    -----------
    gamma : float
        Focusing parameter. Higher values give more weight to hard examples.
    alpha : float
        Weighting factor for the rare/positive class. Higher values give more 
        weight to the rare class.
    size_average : bool
        Whether to average the loss across observations.
    """
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        
    def forward(self, inputs, targets):
        """
        Calculate focal loss.
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Predicted logits of shape [B, C, H, W] for a segmentation task.
        targets : torch.Tensor
            Ground truth labels of shape [B, H, W] with class indices.
            
        Returns:
        --------
        loss : torch.Tensor
            Calculated focal loss.
        """
        # Flatten inputs and targets
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.size(1))
        targets = targets.view(-1)
        
        # Compute standard cross entropy loss
        log_softmax = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(log_softmax, targets, reduction='none')
        
        # Get probabilities for correct class
        softmax = F.softmax(inputs, dim=1)
        pt = softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply focusing parameter
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class balancing
        if self.alpha is not None:
            alpha_weight = torch.ones_like(targets, device=inputs.device)
            alpha_weight[targets == 1] = self.alpha  # Weight for positive class
            alpha_weight[targets == 0] = 1 - self.alpha  # Weight for negative class
            focal_weight = alpha_weight * focal_weight
        
        loss = focal_weight * ce_loss
        
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that focuses on the boundaries between changed and 
    unchanged regions.
    
    This is particularly helpful for detecting small changed areas by 
    emphasizing the importance of boundary pixels.
    """
    def __init__(self, theta=1.0):
        super(BoundaryLoss, self).__init__()
        self.theta = theta
        
    def forward(self, pred, target):
        """
        Calculate boundary-aware loss.
        
        Parameters:
        -----------
        pred : torch.Tensor
            Predicted segmentation logits [B, C, H, W].
        target : torch.Tensor
            Ground truth segmentation [B, H, W].
            
        Returns:
        --------
        loss : torch.Tensor
            Calculated boundary-aware loss.
        """
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        
        # Calculate gradient of target (boundaries)
        target_boundaries = self._compute_boundaries(target_one_hot)
        
        # Calculate standard cross-entropy loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Increase loss weight near boundaries
        boundary_weight = 1.0 + self.theta * target_boundaries
        
        # Apply boundary weights to CE loss
        weighted_loss = boundary_weight * ce_loss
        
        return weighted_loss.mean()
    
    def _compute_boundaries(self, mask):
        """
        Compute boundaries of segmentation mask using gradient.
        
        Parameters:
        -----------
        mask : torch.Tensor
            One-hot encoded mask [B, C, H, W].
            
        Returns:
        --------
        boundaries : torch.Tensor
            Boundary map with higher values at mask edges.
        """
        # Only need to compute gradient for the foreground class (usually index 1)
        foreground = mask[:, 1:2, :, :]
        
        # Compute gradients in x and y directions
        grad_y = torch.abs(foreground[:, :, 1:, :] - foreground[:, :, :-1, :])
        grad_x = torch.abs(foreground[:, :, :, 1:] - foreground[:, :, :, :-1])
        
        # Pad gradients to match input size
        grad_y = F.pad(grad_y, (0, 0, 1, 0), 'constant', 0)
        grad_x = F.pad(grad_x, (1, 0, 0, 0), 'constant', 0)
        
        # Combine gradients
        boundaries = torch.max(grad_y, grad_x).squeeze(1)
        
        return boundaries


class CombinedLoss(nn.Module):
    """
    Combined loss function using Focal Loss for class imbalance and hard examples,
    and Boundary Loss for enhancing detection of small objects/boundaries.
    
    Parameters:
    -----------
    focal_weight : float
        Weight for focal loss component.
    boundary_weight : float
        Weight for boundary loss component.
    dice_weight : float
        Weight for dice loss component.
    gamma : float
        Focusing parameter for focal loss.
    alpha : float
        Class balancing factor for focal loss.
    """
    def __init__(self, focal_weight=1.0, boundary_weight=0.5, dice_weight=0.5, 
                 gamma=2.0, alpha=0.25):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.boundary_loss = BoundaryLoss(theta=3.0)
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        
    def dice_loss(self, pred, target):
        """Calculate Dice loss for smooth boundaries."""
        # Get probabilities for the positive class
        pred = F.softmax(pred, dim=1)[:, 1]
        
        # Convert target to one-hot if it's not
        if target.dim() == 3:
            target = (target == 1).float()
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate Dice coefficient
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        # Add smoothing factor to avoid division by zero
        smooth = 1e-5
        dice = (2. * intersection + smooth) / (union + smooth)
        
        return 1 - dice
        
    def forward(self, pred, target):
        """
        Calculate combined loss.
        
        Parameters:
        -----------
        pred : torch.Tensor
            Predicted segmentation logits [B, C, H, W].
        target : torch.Tensor
            Ground truth segmentation [B, H, W].
            
        Returns:
        --------
        loss : torch.Tensor
            Calculated combined loss.
        """
        fl = self.focal_loss(pred, target)
        bl = self.boundary_loss(pred, target)
        dl = self.dice_loss(pred, target)
        
        return self.focal_weight * fl + self.boundary_weight * bl + self.dice_weight * dl