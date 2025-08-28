import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from typing import Dict, Optional, Tuple
import logging

from ..models.vitallens_emotion import VitalLensEmotionModel
from ..models.loss import VitalLensEmotionLoss
from ..data.dataset import RPPGEmotionDataset
from ..processing.signal_quality import SignalQualityAssessment

class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        self.model_variant = 'vitallens_emotion'
        self.window_size = 150
        self.batch_size = 8
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.scheduler_factor = 0.5
        self.scheduler_patience = 5
        self.patience = 10
        self.min_delta = 0.001
        self.train_split = 0.8
        self.augment_train = True
        self.min_quality = 0.3
        self.val_interval = 1
        self.save_interval = 5
        self.log_interval = 10

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class VitalLensTrainer:
    """Complete VitalLens training pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.experiment_name = f"vitallens_{config.model_variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(f"./logs/{self.experiment_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        
        # Initialize model
        self.model = VitalLensEmotionModel(
            sequence_length=config.window_size,
            num_emotions=7,
            dropout_rate=0.3
        )
        self.model = self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = VitalLensEmotionLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        
        print(f"🚀 Initialized trainer: {self.experiment_name}")
        print(f"   Device: {self.device}")
        print(f"   Model: {config.model_variant}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, dataset_paths):
        """Prepare training and validation datasets"""
        print("📊 Preparing datasets...")
        
        # Use the best available dataset
        if 'ubfc' in dataset_paths and dataset_paths['ubfc'].exists():
            dataset_path = dataset_paths['ubfc']
            dataset_type = 'UBFC-rPPG'
        elif 'sample' in dataset_paths:
            dataset_path = dataset_paths['sample']
            dataset_type = 'SAMPLE'
        else:
            raise ValueError("No suitable dataset found")
        
        print(f"Using dataset: {dataset_type} from {dataset_path}")
        
        # Create full dataset
        full_dataset = RPPGEmotionDataset(
            data_dir=str(dataset_path),
            sequence_length=self.config.window_size,
            transform=None,
            enable_synthetic=True
        )
        
        if len(full_dataset) == 0:
            raise ValueError("No samples passed quality filtering")
        
        # Split dataset
        train_size = int(self.config.train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"✅ Data prepared:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Batch size: {self.config.batch_size}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_hr_loss = 0
        total_emotion_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, batch_data in enumerate(pbar):
            if isinstance(batch_data, dict):
                frames = batch_data['frames']
                targets = {k: v for k, v in batch_data.items() if k != 'frames'}
            else:
                frames, targets = batch_data
                if not isinstance(targets, dict):
                    targets = {'heart_rate': targets}
            
            frames = frames.to(self.device, non_blocking=True)
            for key in targets:
                targets[key] = targets[key].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(frames)
            
            # Compute loss
            loss_dict = self.criterion(predictions, targets)
            total_loss_batch = loss_dict['total_loss']
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_hr_loss += loss_dict.get('hr_loss', 0)
            total_emotion_loss += loss_dict.get('emotion_loss', 0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'HR': f'{loss_dict.get("hr_loss", 0):.4f}',
                'Emotion': f'{loss_dict.get("emotion_loss", 0):.4f}'
            })
            
            # Log to tensorboard
            if batch_idx % self.config.log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', total_loss_batch.item(), step)
                self.writer.add_scalar('Train/HR_Loss', loss_dict.get('hr_loss', 0), step)
                self.writer.add_scalar('Train/Emotion_Loss', loss_dict.get('emotion_loss', 0), step)
        
        # Return average losses
        return {
            'total_loss': total_loss / len(self.train_loader),
            'hr_loss': total_hr_loss / len(self.train_loader),
            'emotion_loss': total_emotion_loss / len(self.train_loader)
        }
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_hr_predictions = []
        all_hr_targets = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc='Validation'):
                if isinstance(batch_data, dict):
                    frames = batch_data['frames']
                    targets = {k: v for k, v in batch_data.items() if k != 'frames'}
                else:
                    frames, targets = batch_data
                    if not isinstance(targets, dict):
                        targets = {'heart_rate': targets}
                
                frames = frames.to(self.device, non_blocking=True)
                for key in targets:
                    targets[key] = targets[key].to(self.device, non_blocking=True)
                
                predictions = self.model(frames)
                loss_dict = self.criterion(predictions, targets)
                
                total_loss += loss_dict['total_loss'].item()
                
                if 'heart_rate' in predictions and 'heart_rate' in targets:
                    all_hr_predictions.extend(predictions['heart_rate'].cpu().numpy())
                    all_hr_targets.extend(targets['heart_rate'].cpu().numpy())
        
        # Calculate metrics
        mae = 0
        rmse = 0
        r2 = 0
        correlation = 0
        
        if all_hr_predictions and all_hr_targets:
            mae = mean_absolute_error(all_hr_targets, all_hr_predictions)
            rmse = np.sqrt(mean_squared_error(all_hr_targets, all_hr_predictions))
            r2 = r2_score(all_hr_targets, all_hr_predictions)
            
            if len(all_hr_targets) > 1:
                correlation, _ = pearsonr(all_hr_targets, all_hr_predictions)
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', total_loss / len(self.val_loader), epoch)
        self.writer.add_scalar('Val/MAE', mae, epoch)
        self.writer.add_scalar('Val/RMSE', rmse, epoch)
        self.writer.add_scalar('Val/R2', r2, epoch)
        self.writer.add_scalar('Val/Correlation', correlation, epoch)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation,
            'predictions': all_hr_predictions,
            'targets': all_hr_targets
        }
    
    def train(self, dataset_paths):
        """Full training loop"""
        print(f"🎯 Starting training: {self.experiment_name}")
        
        # Prepare data
        self.prepare_data(dataset_paths)
        
        # Training loop
        best_mae = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📈 Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['total_loss'])
            
            # Validate
            if epoch % self.config.val_interval == 0:
                val_metrics = self.validate_epoch(epoch)
                self.val_losses.append(val_metrics['loss'])
                self.val_maes.append(val_metrics['mae'])
                
                # Update scheduler
                self.scheduler.step(val_metrics['loss'])
                
                # Print metrics
                print(f"   Train Loss: {train_metrics['total_loss']:.4f}")
                print(f"   Val Loss: {val_metrics['loss']:.4f}")
                print(f"   Val MAE: {val_metrics['mae']:.2f} BPM")
                print(f"   Val RMSE: {val_metrics['rmse']:.2f} BPM")
                print(f"   Val R²: {val_metrics['r2']:.3f}")
                print(f"   Val Correlation: {val_metrics['correlation']:.3f}")
                
                # Save best model
                if val_metrics['mae'] < best_mae:
                    best_mae = val_metrics['mae']
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                
                # Early stopping
                if self.early_stopping(val_metrics['loss'], self.model):
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Save periodic checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch, val_metrics if 'val_metrics' in locals() else None)
        
        print(f"\n✅ Training completed!")
        print(f"   Best MAE: {best_mae:.2f} BPM")
        print(f"   Model saved to: {self.log_dir}")
        
        self.writer.close()
        
        return best_mae
    
    def save_checkpoint(self, epoch, metrics=None, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maes': self.val_maes
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        # Save checkpoint
        checkpoint_path = self.log_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.log_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved: {best_path}")
