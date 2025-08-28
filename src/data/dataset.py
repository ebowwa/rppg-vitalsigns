import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
import os
from typing import Dict, Tuple, Optional
import random
from datasets import load_dataset
import librosa
import soundfile as sf

class RPPGEmotionDataset(Dataset):
    def __init__(self, data_dir, metadata_file, fer2013_dir=None, sequence_length=150, 
                 image_size=(224, 224), augment=True, emotion_prob=0.3,
                 enable_audio=False, enable_eyetracking=False, hf_emotion_datasets=None):
        self.data_dir = Path(data_dir)
        self.fer2013_dir = Path(fer2013_dir) if fer2013_dir else None
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.augment = augment
        self.emotion_prob = emotion_prob
        self.enable_audio = enable_audio
        self.enable_eyetracking = enable_eyetracking
        
        self.metadata = pd.read_csv(metadata_file)
        
        if self.fer2013_dir and self.fer2013_dir.exists():
            self.fer2013_data = self._load_fer2013_data()
        else:
            self.fer2013_data = None
        
        self.hf_emotion_datasets = []
        if hf_emotion_datasets:
            for dataset_name in hf_emotion_datasets:
                try:
                    dataset = load_dataset(dataset_name, split='train')
                    self.hf_emotion_datasets.append(dataset)
                except Exception as e:
                    print(f"Failed to load {dataset_name}: {e}")
        
        self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_fer2013_data(self):
        if self.fer2013_dir is None:
            return None
        fer_csv_path = self.fer2013_dir / 'fer2013.csv'
        if fer_csv_path.exists():
            return pd.read_csv(fer_csv_path)
        return None
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        video_frames = self._generate_synthetic_frames()
        targets = self._generate_synthetic_targets(row)
        
        if self.fer2013_data is not None and np.random.random() < self.emotion_prob:
            emotion_label = self._get_fer2013_emotion()
        else:
            emotion_label = np.random.randint(0, 7)
        
        targets['emotion_labels'] = torch.LongTensor([emotion_label])
        
        if self.enable_audio:
            audio_features = self._generate_synthetic_audio()
            targets['audio_emotion_labels'] = torch.LongTensor([emotion_label])
        
        if self.enable_eyetracking:
            eyetrack_coords = self._generate_synthetic_eyetracking()
            targets['eyetrack_targets'] = torch.FloatTensor(eyetrack_coords)
        
        result = [video_frames, targets]
        
        if self.enable_audio:
            result.append(audio_features)
        
        if self.enable_eyetracking:
            result.append(eyetrack_coords)
        
        if len(result) == 2:
            return video_frames, targets
        else:
            return tuple(result)
    
    def _generate_synthetic_frames(self):
        frames = []
        for i in range(self.sequence_length):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            pulse_signal = 0.1 * np.sin(2 * np.pi * i / 30)
            frame = np.clip(frame + pulse_signal * 10, 0, 255).astype(np.uint8)
            
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)
        
        return torch.stack(frames)
    
    def _generate_synthetic_targets(self, row):
        hr = np.random.uniform(60, 100)
        pulse_freq = hr / 60.0
        time_points = np.linspace(0, 5, self.sequence_length)
        pulse_waveform = np.sin(2 * np.pi * pulse_freq * time_points)
        pulse_waveform += 0.1 * np.random.randn(self.sequence_length)
        
        rr = np.random.uniform(12, 20)
        resp_freq = rr / 60.0
        resp_waveform = 0.5 * np.sin(2 * np.pi * resp_freq * time_points)
        resp_waveform += 0.05 * np.random.randn(self.sequence_length)
        
        return {
            'pulse_waveform': torch.FloatTensor(pulse_waveform),
            'resp_waveform': torch.FloatTensor(resp_waveform),
            'heart_rate': torch.FloatTensor([hr]),
            'resp_rate': torch.FloatTensor([rr]),
            'subject_age': torch.FloatTensor([row.get('subject_age', 30)]),
            'subject_gender': torch.LongTensor([1 if row.get('subject_gender', 'male') == 'male' else 0]),
            'subject_skin_type': torch.LongTensor([row.get('subject_skin_type', 3)])
        }
    
    def _get_fer2013_emotion(self):
        if self.fer2013_data is None:
            return np.random.randint(0, 7)
        
        random_idx = np.random.randint(0, len(self.fer2013_data))
        return self.fer2013_data.iloc[random_idx]['emotion']
    
    def _generate_synthetic_audio(self) -> torch.Tensor:
        sample_rate = 16000
        duration = 3.0
        audio_signal = np.random.randn(int(sample_rate * duration)) * 0.1
        
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_signal, sr=sample_rate, n_mels=128, fmax=8000
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        mel_tensor = torch.tensor(mel_spectrogram_db, dtype=torch.float32).unsqueeze(0)
        
        if mel_tensor.shape[-1] < 128:
            padding = 128 - mel_tensor.shape[-1]
            mel_tensor = torch.nn.functional.pad(mel_tensor, (0, padding))
        elif mel_tensor.shape[-1] > 128:
            mel_tensor = mel_tensor[:, :, :128]
        
        return mel_tensor
    
    def _generate_synthetic_eyetracking(self) -> np.ndarray:
        gaze_x = np.random.uniform(-1, 1)
        gaze_y = np.random.uniform(-1, 1)
        return np.array([gaze_x, gaze_y])
