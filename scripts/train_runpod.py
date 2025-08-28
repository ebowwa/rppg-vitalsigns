import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.vitallens_emotion import VitalLensEmotionModel
from src.models.loss import VitalLensEmotionLoss
from src.data.dataset import RPPGEmotionDataset
from src.utils.metrics import calculate_rppg_metrics, calculate_emotion_metrics
from datasets import load_dataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    wandb.init(project="vitallens-emotion", name="runpod-training")
    
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    synthetic_metadata = pd.DataFrame({
        'chunk_id': range(1000),
        'subject_age': np.random.randint(18, 80, 1000),
        'subject_gender': np.random.choice(['male', 'female'], 1000),
        'subject_skin_type': np.random.randint(1, 7, 1000),
        'frame_avg_hr_pox': np.random.uniform(60, 100, 1000),
        'frame_avg_rr': np.random.uniform(12, 20, 1000)
    })
    
    metadata_path = data_dir / 'synthetic_training_data.csv'
    synthetic_metadata.to_csv(metadata_path, index=False)
    
    hf_datasets = [
        'AdamCodd/yolo-emotions',
        'HazemAbdelkawy/emotions',
        'ChristophSchuhmann/emotions'
    ]
    
    train_dataset = RPPGEmotionDataset(
        data_dir=data_dir / 'synthetic_data',
        metadata_file=metadata_path,
        fer2013_dir=data_dir / 'fer2013',
        sequence_length=150,
        augment=True,
        enable_audio=True,
        enable_eyetracking=True,
        hf_emotion_datasets=hf_datasets
    )
    
    val_dataset = RPPGEmotionDataset(
        data_dir=data_dir / 'synthetic_data',
        metadata_file=metadata_path,
        fer2013_dir=data_dir / 'fer2013',
        sequence_length=150,
        augment=False,
        enable_audio=True,
        enable_eyetracking=True,
        hf_emotion_datasets=hf_datasets
    )
    
    batch_size = 4 if device.type == 'cpu' else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = VitalLensEmotionModel(
        sequence_length=150, 
        enable_audio=True, 
        enable_eyetracking=True
    ).to(device)
    criterion = VitalLensEmotionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_rppg_metrics = {}
        train_emotion_metrics = {}
        
        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 2:
                video_frames, targets = batch_data
                audio_features = None
                eyetrack_coords = None
            else:
                video_frames, targets, audio_features, eyetrack_coords = batch_data
            
            video_frames = video_frames.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            if audio_features is not None:
                audio_features = audio_features.to(device)
            if eyetrack_coords is not None:
                eyetrack_coords = eyetrack_coords.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                predictions = model(video_frames, audio_features, eyetrack_coords)
                loss, loss_dict = criterion(predictions, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                batch_rppg_metrics = calculate_rppg_metrics(predictions, targets)
                batch_emotion_metrics = calculate_emotion_metrics(predictions, targets)
                
                for key, value in batch_rppg_metrics.items():
                    if key not in train_rppg_metrics:
                        train_rppg_metrics[key] = 0
                    train_rppg_metrics[key] += value
                
                for key, value in batch_emotion_metrics.items():
                    if key not in train_emotion_metrics:
                        train_emotion_metrics[key] = 0
                    if key == 'emotion_accuracy':
                        train_emotion_metrics[key] += value
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_rppg_metrics = {k: v / len(train_loader) for k, v in train_rppg_metrics.items()}
        avg_train_emotion_acc = train_emotion_metrics.get('emotion_accuracy', 0) / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_rppg_metrics = {}
        val_emotion_metrics = {}
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 2:
                    video_frames, targets = batch_data
                    audio_features = None
                    eyetrack_coords = None
                else:
                    video_frames, targets, audio_features, eyetrack_coords = batch_data
                
                video_frames = video_frames.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                if audio_features is not None:
                    audio_features = audio_features.to(device)
                if eyetrack_coords is not None:
                    eyetrack_coords = eyetrack_coords.to(device)
                
                predictions = model(video_frames, audio_features, eyetrack_coords)
                loss, loss_dict = criterion(predictions, targets)
                val_loss += loss.item()
                
                batch_rppg_metrics = calculate_rppg_metrics(predictions, targets)
                batch_emotion_metrics = calculate_emotion_metrics(predictions, targets)
                
                for key, value in batch_rppg_metrics.items():
                    if key not in val_rppg_metrics:
                        val_rppg_metrics[key] = 0
                    val_rppg_metrics[key] += value
                
                for key, value in batch_emotion_metrics.items():
                    if key not in val_emotion_metrics:
                        val_emotion_metrics[key] = 0
                    if key == 'emotion_accuracy':
                        val_emotion_metrics[key] += value
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_rppg_metrics = {k: v / len(val_loader) for k, v in val_rppg_metrics.items()}
        avg_val_emotion_acc = val_emotion_metrics.get('emotion_accuracy', 0) / len(val_loader)
        
        scheduler.step()
        
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_hr_mae': avg_train_rppg_metrics.get('hr_mae', 0),
            'val_hr_mae': avg_val_rppg_metrics.get('hr_mae', 0),
            'train_emotion_acc': avg_train_emotion_acc,
            'val_emotion_acc': avg_val_emotion_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train HR MAE: {avg_train_rppg_metrics.get('hr_mae', 0):.2f}, Val HR MAE: {avg_val_rppg_metrics.get('hr_mae', 0):.2f}")
        print(f"Train Emotion Acc: {avg_train_emotion_acc:.3f}, Val Emotion Acc: {avg_val_emotion_acc:.3f}")
        print("-" * 50)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_hr_mae': avg_val_rppg_metrics.get('hr_mae', 0),
                'val_emotion_acc': avg_val_emotion_acc,
                'model_config': {
                    'sequence_length': 150,
                    'num_emotions': 7,
                    'dropout_rate': 0.3
                }
            }
            torch.save(checkpoint, 'best_model.pth')
            print(f"Saved best model with val_loss: {avg_val_loss:.4f}")
    
    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    main()
