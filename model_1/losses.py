import torch
import torch.nn as nn
import torch.nn.functional as F

from model_1.config import Config

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss to address extreme class imbalances.
    Focal loss down-weights easy-to-classify samples and focuses training
    on hard samples (e.g. noisy variability vs true transits).
    
    Formula:
      FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        # alpha is expected to be a tensor of weights for each class
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_device = self.alpha.to(inputs.device)
            focal_loss = focal_loss * alpha_device[targets]
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Computes total multi-task loss combining classification, regression, and confidence heads.
    
    Research Enhancement: Masked Regression Loss.
    We only compute regression losses for true exoplanet transit samples (label index 0).
    Calculating orbital parameters (like depth/duration) on instrument noise or eclipsing binaries 
    leads to gradient corruption. Masking guarantees regression heads only train on valid transits.
    """
    def __init__(self, config=Config, class_weights=None, use_focal=True):
        super().__init__()
        self.config = config
        
        if use_focal:
            self.class_loss_fn = FocalLoss(gamma=2.0, alpha=class_weights)
        else:
            self.class_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            
        self.reg_loss_fn = nn.MSELoss(reduction='none')  # We will manually mask and mean-reduce
        self.conf_loss_fn = nn.BCELoss()

    def forward(self, class_logits, reg_outputs, confidence, targets_class, targets_reg, targets_conf):
        """
        Args:
            class_logits: (B, NUM_CLASSES)
            reg_outputs: (B, 4) -> [depth, duration, period, midpoint]
            confidence: (B, 1)
            targets_class: (B)
            targets_reg: (B, 4)
            targets_conf: (B, 1)
        """
        # 1. Classification Loss (Softmax multi-class)
        loss_class = self.class_loss_fn(class_logits, targets_class)

        # 2. Masked Regression Loss
        # Mask: True only for samples belonging to Exoplanet Transit (class index 0)
        transit_mask = (targets_class == 0)
        
        if transit_mask.sum() > 0:
            # Slices of inputs for transits only
            transit_reg_pred = reg_outputs[transit_mask]
            transit_reg_true = targets_reg[transit_mask]
            
            # Compute element-wise MSE
            mse_elements = self.reg_loss_fn(transit_reg_pred, transit_reg_true)
            
            # Reduce per parameter
            loss_depth = mse_elements[:, 0].mean()
            loss_duration = mse_elements[:, 1].mean()
            loss_period = mse_elements[:, 2].mean()
            loss_midpoint = mse_elements[:, 3].mean()
        else:
            # Fallback if no transits in batch to prevent NaN gradient
            loss_depth = torch.tensor(0.0, device=reg_outputs.device)
            loss_duration = torch.tensor(0.0, device=reg_outputs.device)
            loss_period = torch.tensor(0.0, device=reg_outputs.device)
            loss_midpoint = torch.tensor(0.0, device=reg_outputs.device)

        # 3. Confidence Head Loss (BCE)
        loss_conf = self.conf_loss_fn(confidence, targets_conf)

        # 4. Weighted Total Loss
        total_loss = (
            self.config.LAMBDA_CLASSIFICATION * loss_class +
            self.config.LAMBDA_DEPTH * loss_depth +
            self.config.LAMBDA_DURATION * loss_duration +
            self.config.LAMBDA_PERIOD * loss_period +
            self.config.LAMBDA_MIDPOINT * loss_midpoint +
            self.config.LAMBDA_CONFIDENCE * loss_conf
        )

        return {
            'loss': total_loss,
            'class_loss': loss_class,
            'depth_loss': loss_depth,
            'duration_loss': loss_duration,
            'period_loss': loss_period,
            'midpoint_loss': loss_midpoint,
            'conf_loss': loss_conf
        }
