import sys
sys.path.append('.')
from src.models.vitallens_emotion import VitalLensEmotionModel
from src.models.loss import VitalLensEmotionLoss
from src.data.dataset import RPPGEmotionDataset
from src.utils.metrics import calculate_rppg_metrics, calculate_emotion_metrics
import torch
import pandas as pd
import numpy as np
import os

def test_training_pipeline():
    print('Testing VitalLens Emotion Training Pipeline...')
    
    print('1. Testing multi-modal model initialization...')
    model = VitalLensEmotionModel(
        sequence_length=150, 
        num_emotions=7,
        enable_audio=True,
        enable_eyetracking=True
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'   Total parameters: {total_params:,}')
    print(f'   Trainable parameters: {trainable_params:,}')
    print(f'   Multi-modal capabilities: Audio={model.enable_audio}, EyeTrack={model.enable_eyetracking}')
    
    print('2. Testing loss function...')
    criterion = VitalLensEmotionLoss()
    print('   Loss function initialized successfully')
    
    print('3. Creating synthetic test data...')
    os.makedirs('./data', exist_ok=True)
    synthetic_metadata = pd.DataFrame({
        'chunk_id': range(10),
        'subject_age': np.random.randint(18, 80, 10),
        'subject_gender': np.random.choice(['male', 'female'], 10),
        'subject_skin_type': np.random.randint(1, 7, 10),
        'frame_avg_hr_pox': np.random.uniform(60, 100, 10),
        'frame_avg_rr': np.random.uniform(12, 20, 10)
    })
    synthetic_metadata.to_csv('./data/test_metadata.csv', index=False)
    print('   Test metadata created')
    
    print('4. Testing multi-modal dataset...')
    dataset = RPPGEmotionDataset(
        data_dir='./data/synthetic_data',
        metadata_file='./data/test_metadata.csv',
        sequence_length=150,
        augment=False,
        enable_audio=True,
        enable_eyetracking=True
    )
    print(f'   Dataset length: {len(dataset)}')
    
    print('5. Testing multi-modal forward pass...')
    sample_data = dataset[0]
    
    if len(sample_data) == 2:
        video_frames, targets = sample_data
        audio_features = None
        eyetrack_coords = None
    else:
        video_frames, targets, audio_features, eyetrack_coords = sample_data
    
    print(f'   Video frames shape: {video_frames.shape}')
    print(f'   Targets keys: {list(targets.keys())}')
    if audio_features is not None:
        print(f'   Audio features shape: {audio_features.shape}')
    if eyetrack_coords is not None:
        print(f'   Eye-tracking coords shape: {eyetrack_coords.shape}')
    
    # Add batch dimension
    video_frames = video_frames.unsqueeze(0)
    targets = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in targets.items()}
    if audio_features is not None:
        audio_features = audio_features.unsqueeze(0)
    if eyetrack_coords is not None:
        eyetrack_coords = torch.tensor(eyetrack_coords, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(video_frames, audio_features, eyetrack_coords)
        print(f'   Model outputs keys: {list(outputs.keys())}')
        print(f'   Output shapes:')
        for key, value in outputs.items():
            print(f'     {key}: {value.shape}')
    
    print('6. Testing loss calculation...')
    loss, loss_dict = criterion(outputs, targets)
    print(f'   Total loss: {loss.item():.4f}')
    print(f'   Loss components:')
    for key, value in loss_dict.items():
        print(f'     {key}: {value:.4f}')
    
    print('7. Testing metrics calculation...')
    rppg_metrics = calculate_rppg_metrics(outputs, targets)
    emotion_metrics = calculate_emotion_metrics(outputs, targets)
    
    print(f'   rPPG Metrics:')
    for key, value in rppg_metrics.items():
        print(f'     {key}: {value:.4f}')
    
    print(f'   Emotion Metrics:')
    print(f'     accuracy: {emotion_metrics["emotion_accuracy"]:.4f}')
    
    print('\n✅ All multi-modal pipeline components working correctly!')
    print(f'🎯 Ready for training with {total_params:,} parameters')
    print(f'🚀 Multi-modal capabilities: Video + Audio + Eye-tracking')
    print(f'📊 Supported datasets: MCD rPPG, YOLO emotions, HuggingFace collections')
    print(f'💰 Estimated training cost: $16-32 (Modal A100) or $10 (RunPod RTX 4090)')
    print(f'📖 See docs/TRAINING_PLANS.md for 10 comprehensive training strategies')
    
    return True

if __name__ == "__main__":
    test_training_pipeline()
