import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


from model_1.config import Config
from model_1.losses import MultiTaskLoss

class EarlyStopping:
    """
    Early stopping to stop the training when validation loss stops improving.
    """
    def __init__(self, patience=8, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ExoplanetTrainer:
    """
    Manages the PyTorch training, validation, checkpoints, and schedules.
    Implements:
      - Mixed Precision Training (AMP)
      - Cosine Annealing LR scheduler
      - Gradient Clipping
      - Early Stopping
    """
    def __init__(self, model, train_loader, val_loader, class_weights=None, config=Config, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Select device
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model.to(self.device)
        
        # Multi-task loss
        self.loss_fn = MultiTaskLoss(config=config, class_weights=class_weights, use_focal=True)
        self.loss_fn.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.EPOCHS, 
            eta_min=1e-6
        )
        
        # Mixed precision gradient scaler
        is_cuda = (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.MIXED_PRECISION and is_cuda)
        
        # Early Stopping
        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True)
        
        # Training history logs
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_class_acc': [], 'val_class_acc': [],
            'train_class_loss': [], 'val_class_loss': [],
            'train_reg_loss': [], 'val_reg_loss': [],
            'train_conf_loss': [], 'val_conf_loss': []
        }

    def train_epoch(self):
        self.model.train()
        running_losses = {
            'loss': 0.0, 'class_loss': 0.0, 'depth_loss': 0.0,
            'duration_loss': 0.0, 'period_loss': 0.0, 'midpoint_loss': 0.0, 'conf_loss': 0.0
        }
        
        correct_class = 0
        total_samples = 0
        
        for batch_idx, (x, y_class, y_reg, y_conf) in enumerate(self.train_loader):
            # Move to target device
            x = x.to(self.device)
            y_class = y_class.to(self.device)
            y_reg = y_reg.to(self.device)
            y_conf = y_conf.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass under autocast (mixed precision)
            device_type = self.device.type
            with torch.amp.autocast(device_type=device_type, enabled=self.config.MIXED_PRECISION and device_type in ['cuda', 'cpu']):
                class_logits, reg_outputs, confidence, _ = self.model(x)
                
                # Compute multi-task loss
                loss_dict = self.loss_fn(
                    class_logits, reg_outputs, confidence,
                    y_class, y_reg, y_conf
                )
                loss = loss_dict['loss']
            
            # Backward pass and optimization
            self.scaler.scale(loss).backward()
            
            # Gradient clipping to prevent gradient explosions in LSTMs
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.GRADIENT_CLIPPING)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Record losses
            for k in running_losses.keys():
                running_losses[k] += loss_dict[k].item()
                
            # Accuracy metric
            _, preds = torch.max(class_logits, 1)
            correct_class += (preds == y_class).sum().item()
            total_samples += y_class.size(0)

        # Average results
        num_batches = len(self.train_loader)
        avg_losses = {k: v / num_batches for k, v in running_losses.items()}
        class_acc = correct_class / total_samples
        
        return avg_losses, class_acc

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_losses = {
            'loss': 0.0, 'class_loss': 0.0, 'depth_loss': 0.0,
            'duration_loss': 0.0, 'period_loss': 0.0, 'midpoint_loss': 0.0, 'conf_loss': 0.0
        }
        
        correct_class = 0
        total_samples = 0
        
        for x, y_class, y_reg, y_conf in self.val_loader:
            x = x.to(self.device)
            y_class = y_class.to(self.device)
            y_reg = y_reg.to(self.device)
            y_conf = y_conf.to(self.device)
            
            # Forward pass
            class_logits, reg_outputs, confidence, _ = self.model(x)
            loss_dict = self.loss_fn(
                class_logits, reg_outputs, confidence,
                y_class, y_reg, y_conf
            )
            
            # Record losses
            for k in running_losses.keys():
                running_losses[k] += loss_dict[k].item()
                
            # Accuracy metric
            _, preds = torch.max(class_logits, 1)
            correct_class += (preds == y_class).sum().item()
            total_samples += y_class.size(0)

        num_batches = len(self.val_loader)
        avg_losses = {k: v / num_batches for k, v in running_losses.items()}
        class_acc = correct_class / total_samples
        
        return avg_losses, class_acc

    def fit(self):
        print(f"Starting training on device: {self.device}")
        print(f"Option: {'Option B (BLS Transit Window)' if self.config.USE_BLS_WINDOWS else 'Option A (Full Light Curve)'}")
        print("="*60)
        
        best_val_loss = float('inf')
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, "best_model.pt")
        
        for epoch in range(1, self.config.EPOCHS + 1):
            # Train and validate epoch
            train_losses, train_acc = self.train_epoch()
            val_losses, val_acc = self.validate()
            
            # Step LR scheduler
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_losses['loss'])
            self.history['val_loss'].append(val_losses['loss'])
            self.history['train_class_acc'].append(train_acc)
            self.history['val_class_acc'].append(val_acc)
            
            self.history['train_class_loss'].append(train_losses['class_loss'])
            self.history['val_class_loss'].append(val_losses['class_loss'])
            
            # Sum regression components for combined regression metric
            train_reg = (train_losses['depth_loss'] + train_losses['duration_loss'] + 
                         train_losses['period_loss'] + train_losses['midpoint_loss'])
            val_reg = (val_losses['depth_loss'] + val_losses['duration_loss'] + 
                       val_losses['period_loss'] + val_losses['midpoint_loss'])
            self.history['train_reg_loss'].append(train_reg)
            self.history['val_reg_loss'].append(val_reg)
            
            self.history['train_conf_loss'].append(train_losses['conf_loss'])
            self.history['val_conf_loss'].append(val_losses['conf_loss'])
            
            # Print metrics
            print(f"Epoch {epoch:02d}/{self.config.EPOCHS:02d} | "
                  f"Train Loss: {train_losses['loss']:.4f} (Acc: {train_acc:.2%}) | "
                  f"Val Loss: {val_losses['loss']:.4f} (Acc: {val_acc:.2%})")
            
            # Checkpoint saving
            if val_losses['loss'] < best_val_loss:
                best_val_loss = val_losses['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'val_acc': val_acc
                }, checkpoint_path)
                print(f"--> Saved best model checkpoint to {checkpoint_path}")
                
            # Early Stopping check
            self.early_stopping(val_losses['loss'])
            if self.early_stopping.early_stop:
                print("Early stopping triggered. Training stopped.")
                break
                
        # Load best weights before returning
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best weights from epoch {checkpoint['epoch']} with Val Loss: {checkpoint['val_loss']:.4f}")
        
        return self.history
