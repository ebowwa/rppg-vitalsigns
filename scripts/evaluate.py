import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.vitallens_emotion import VitalLensEmotionModel
from src.data.dataset import RPPGEmotionDataset
from src.utils.metrics import calculate_rppg_metrics, calculate_emotion_metrics

def evaluate_model(model_path, data_dir="./data"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config', {
        'sequence_length': 150,
        'num_emotions': 7,
        'dropout_rate': 0.3
    })
    
    model = VitalLensEmotionModel(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    data_path = Path(data_dir)
    metadata_path = data_path / 'synthetic_training_data.csv'
    
    if not metadata_path.exists():
        synthetic_metadata = pd.DataFrame({
            'chunk_id': range(100),
            'subject_age': np.random.randint(18, 80, 100),
            'subject_gender': np.random.choice(['male', 'female'], 100),
            'subject_skin_type': np.random.randint(1, 7, 100),
            'frame_avg_hr_pox': np.random.uniform(60, 100, 100),
            'frame_avg_rr': np.random.uniform(12, 20, 100)
        })
        synthetic_metadata.to_csv(metadata_path, index=False)
    
    test_dataset = RPPGEmotionDataset(
        data_dir=data_path / 'synthetic_data',
        metadata_file=metadata_path,
        fer2013_dir=data_path / 'fer2013',
        sequence_length=model_config['sequence_length'],
        augment=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    all_rppg_metrics = []
    all_emotion_preds = []
    all_emotion_true = []
    
    with torch.no_grad():
        for video_frames, targets in test_loader:
            video_frames = video_frames.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            predictions = model(video_frames)
            
            rppg_metrics = calculate_rppg_metrics(predictions, targets)
            emotion_metrics = calculate_emotion_metrics(predictions, targets)
            
            all_rppg_metrics.append(rppg_metrics)
            all_emotion_preds.extend(emotion_metrics['emotion_pred'])
            all_emotion_true.extend(emotion_metrics['emotion_true'])
    
    avg_rppg_metrics = {}
    for key in all_rppg_metrics[0].keys():
        avg_rppg_metrics[key] = np.mean([m[key] for m in all_rppg_metrics])
    
    emotion_accuracy = np.mean(np.array(all_emotion_preds) == np.array(all_emotion_true))
    
    print("=== Evaluation Results ===")
    print(f"Heart Rate MAE: {avg_rppg_metrics['hr_mae']:.2f} BPM")
    print(f"Respiratory Rate MAE: {avg_rppg_metrics['rr_mae']:.2f} BPM")
    print(f"Pulse SNR: {avg_rppg_metrics['pulse_snr']:.2f} dB")
    print(f"Respiration SNR: {avg_rppg_metrics['resp_snr']:.2f} dB")
    print(f"Pulse Correlation: {avg_rppg_metrics['pulse_cor']:.3f}")
    print(f"Respiration Correlation: {avg_rppg_metrics['resp_cor']:.3f}")
    print(f"Emotion Accuracy: {emotion_accuracy:.3f}")
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    print("\n=== Emotion Classification Report ===")
    print(classification_report(all_emotion_true, all_emotion_preds, 
                              target_names=emotion_labels, zero_division=0))
    
    cm = confusion_matrix(all_emotion_true, all_emotion_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title('Emotion Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('emotion_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    results = {
        'rppg_metrics': avg_rppg_metrics,
        'emotion_accuracy': emotion_accuracy,
        'emotion_predictions': all_emotion_preds,
        'emotion_true': all_emotion_true
    }
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate VitalLens Emotion Model')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to the data directory')
    
    args = parser.parse_args()
    
    results = evaluate_model(args.model_path, args.data_dir)
