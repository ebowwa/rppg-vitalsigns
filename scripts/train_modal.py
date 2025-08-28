import modal
import os
from pathlib import Path

app = modal.App("vitallens-emotion-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1),
    timeout=3600 * 4,
    volumes={"/data": modal.Volume.from_name("vitallens-data", create_if_missing=True)}
)
def train_vitallens():
    import sys
    sys.path.append("/root")
    
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import wandb
    import numpy as np
    from pathlib import Path
    
    from src.models.vitallens_emotion import VitalLensEmotionModel
    from src.models.loss import VitalLensEmotionLoss
    from src.data.dataset import RPPGEmotionDataset
    from src.utils.metrics import calculate_rppg_metrics, calculate_emotion_metrics
    from datasets import load_dataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    wandb.init(project="vitallens-emotion", name="modal-training")
    
    import pandas as pd
    
    synthetic_metadata = pd.DataFrame({
        'chunk_id': range(1000),
        'subject_age': np.random.randint(18, 80, 1000),
        'subject_gender': np.random.choice(['male', 'female'], 1000),
        'subject_skin_type': np.random.randint(1, 7, 1000),
        'frame_avg_hr_pox': np.random.uniform(60, 100, 1000),
        'frame_avg_rr': np.random.uniform(12, 20, 1000)
    })
    
    synthetic_metadata.to_csv('/data/synthetic_training_data.csv', index=False)
    
    hf_datasets = [
        'AdamCodd/yolo-emotions',
        'HazemAbdelkawy/emotions',
        'ChristophSchuhmann/emotions'
    ]
    
    train_dataset = RPPGEmotionDataset(
        data_dir='/data/synthetic_data',
        metadata_file='/data/synthetic_training_data.csv',
        fer2013_dir='/data/fer2013',
        sequence_length=150,
        augment=True,
        enable_audio=True,
        enable_eyetracking=True,
        hf_emotion_datasets=hf_datasets
    )
    
    val_dataset = RPPGEmotionDataset(
        data_dir='/data/synthetic_data',
        metadata_file='/data/synthetic_training_data.csv',
        fer2013_dir='/data/fer2013',
        sequence_length=150,
        augment=False,
        enable_audio=True,
        enable_eyetracking=True,
        hf_emotion_datasets=hf_datasets
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    model = VitalLensEmotionModel(
        sequence_length=150, 
        enable_audio=True, 
        enable_eyetracking=True
    ).to(device)
    criterion = VitalLensEmotionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_rppg_metrics = {}
        train_emotion_metrics = {}
        
        for batch_idx, (video_frames, targets) in enumerate(train_loader):
            video_frames = video_frames.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            predictions = model(video_frames)
            
            loss, loss_dict = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
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
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_rppg_metrics = {k: v / len(train_loader) for k, v in train_rppg_metrics.items()}
        avg_train_emotion_acc = train_emotion_metrics.get('emotion_accuracy', 0) / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_rppg_metrics = {}
        val_emotion_metrics = {}
        
        with torch.no_grad():
            for video_frames, targets in val_loader:
                video_frames = video_frames.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                predictions = model(video_frames)
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
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train HR MAE: {avg_train_rppg_metrics.get('hr_mae', 0):.2f}, Val HR MAE: {avg_val_rppg_metrics.get('hr_mae', 0):.2f}")
        print(f"Train Emotion Acc: {avg_train_emotion_acc:.3f}, Val Emotion Acc: {avg_val_emotion_acc:.3f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_hr_mae': avg_val_rppg_metrics.get('hr_mae', 0),
                'val_emotion_acc': avg_val_emotion_acc
            }, '/data/best_model.pth')
    
    wandb.finish()
    print("Training completed!")
    
    auto_deploy = os.environ.get('AUTO_DEPLOY_MOBILE', 'false').lower() == 'true'
    if auto_deploy:
        print("🚀 Auto-triggering mobile deployment...")
        try:
            import subprocess
            subprocess.run([
                'python', 'scripts/deploy_mobile.py',
                '--checkpoint', 'best_model.pth',
                '--output-dir', './mobile_deployment',
                '--model-name', 'VitalLensMultiModal',
                '--target-size-mb', '20',
                '--target-inference-ms', '18',
                '--enable-pruning',
                '--enable-quantization'
            ], check=True)
            print("✅ Mobile deployment completed automatically!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Auto mobile deployment failed: {e}")
        except Exception as e:
            print(f"❌ Auto mobile deployment error: {e}")

if __name__ == "__main__":
    with app.run():
        train_vitallens.remote()
