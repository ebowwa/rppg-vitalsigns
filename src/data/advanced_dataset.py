import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from typing import Dict, List, Optional, Tuple

from ..processing.face_detection import FaceDetectionProcessor
from ..processing.signal_quality import SignalQualityAssessment

class AdvancedRPPGDataset(Dataset):
    """Advanced rPPG dataset with proper preprocessing"""
    
    def __init__(self, data_dir, dataset_type='UBFC-rPPG', window_size=150, 
                 overlap=0.5, min_quality=0.3, augment=False):
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.window_size = window_size
        self.overlap = overlap
        self.min_quality = min_quality
        self.augment = augment
        
        # Initialize processors
        self.face_detector = FaceDetectionProcessor()
        self.quality_assessor = SignalQualityAssessment()
        
        # Load and preprocess data
        self.samples = self._load_and_preprocess_data()
        
        # Data transforms
        self.transform = self._get_transforms()
        
        print(f"✅ Loaded {len(self.samples)} high-quality samples from {dataset_type}")
    
    def _get_transforms(self):
        """Get data augmentation transforms"""
        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if self.augment:
            # Add augmentations (careful not to affect rPPG signal)
            augment_transforms = [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            return transforms.Compose(augment_transforms)
        
        return transforms.Compose(base_transforms)
    
    def _load_and_preprocess_data(self):
        """Load and preprocess video data with quality filtering"""
        samples = []
        
        if self.dataset_type == 'UBFC-rPPG' or self.dataset_type == 'SAMPLE':
            samples = self._process_ubfc_format()
        elif self.dataset_type == 'PURE':
            samples = self._process_pure_format()
        elif self.dataset_type == 'COHFACE':
            samples = self._process_cohface_format()
        
        # Filter by quality
        print(f"Filtering {len(samples)} samples by quality (min_quality={self.min_quality})...")
        high_quality_samples = []
        
        for sample in tqdm(samples[:50]):  # Limit for demo
            quality = self._assess_sample_quality(sample)
            if quality >= self.min_quality:
                sample['quality'] = quality
                high_quality_samples.append(sample)
        
        print(f"Kept {len(high_quality_samples)}/{len(samples)} high-quality samples")
        return high_quality_samples
    
    def _process_ubfc_format(self):
        """Process UBFC-rPPG format data"""
        samples = []
        
        subject_dirs = list(self.data_dir.glob('subject_*'))
        if not subject_dirs:
            print(f"No subject directories found in {self.data_dir}")
            return samples
        
        for subject_dir in subject_dirs:
            video_path = subject_dir / 'vid.avi'
            gt_path = subject_dir / 'ground_truth.txt'
            
            if not (video_path.exists() and gt_path.exists()):
                continue
            
            # Load ground truth
            try:
                gt_bpm = np.loadtxt(gt_path)
            except:
                continue
            
            # Get video info
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if frame_count < self.window_size:
                continue
            
            # Create sliding windows
            step_size = int(self.window_size * (1 - self.overlap))
            
            for start_frame in range(0, frame_count - self.window_size + 1, step_size):
                end_frame = start_frame + self.window_size
                
                # Get corresponding ground truth BPM
                if len(gt_bpm) > 1:
                    gt_start_idx = int(start_frame * len(gt_bpm) / frame_count)
                    gt_end_idx = int(end_frame * len(gt_bpm) / frame_count)
                    window_bpm = np.mean(gt_bpm[gt_start_idx:gt_end_idx])
                else:
                    window_bpm = gt_bpm.item() if np.isscalar(gt_bpm) else gt_bpm[0]
                
                # Skip unrealistic BPM values
                if not (40 <= window_bpm <= 200):
                    continue
                
                samples.append({
                    'video_path': str(video_path),
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'bpm': window_bpm,
                    'fps': fps,
                    'subject_id': subject_dir.name
                })
        
        return samples
    
    def _process_pure_format(self):
        """Process PURE dataset format"""
        # Placeholder - implement based on PURE dataset structure
        return []
    
    def _process_cohface_format(self):
        """Process COHFACE dataset format"""
        # Placeholder - implement based on COHFACE dataset structure
        return []
    
    def _assess_sample_quality(self, sample):
        """Assess quality of a video sample"""
        try:
            # Load a few frames to assess quality
            cap = cv2.VideoCapture(sample['video_path'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample['start_frame'])
            
            rgb_signals = [[], [], []]
            face_detections = 0
            
            # Sample every 10th frame for efficiency
            for i in range(0, min(30, sample['end_frame'] - sample['start_frame']), 10):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect face
                face_info = self.face_detector.detect_face(frame)
                if face_info is not None:
                    face_detections += 1
                    
                    roi = self.face_detector.extract_roi(frame, 'forehead')
                    if roi is not None:
                        rgb_signals[0].append(np.mean(roi[:, :, 2]))  # R
                        rgb_signals[1].append(np.mean(roi[:, :, 1]))  # G
                        rgb_signals[2].append(np.mean(roi[:, :, 0]))  # B
            
            cap.release()
            
            # Calculate quality metrics
            if len(rgb_signals[0]) < 5:  # Need minimum samples
                return 0.0
            
            # Face detection rate
            face_detection_rate = face_detections / min(30, sample['end_frame'] - sample['start_frame']) * 10
            
            if len(rgb_signals[1]) > 0:
                signal_quality = self.quality_assessor.overall_quality_score(np.array(rgb_signals[1]))
                overall_quality = signal_quality['overall_score']
            else:
                overall_quality = 0.0
            
            # Combine metrics
            combined_quality = 0.6 * overall_quality + 0.4 * face_detection_rate
            
            return combined_quality
            
        except Exception as e:
            print(f"Error assessing sample quality: {e}")
            return 0.0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample = self.samples[idx]
        
        # Load video frames
        frames = self._load_video_frames(sample)
        
        if frames is None:
            frames = torch.zeros(self.window_size, 3, 224, 224)
            bpm = 70.0  # Default BPM
        else:
            bpm = sample['bpm']
        
        return frames, torch.tensor(bpm, dtype=torch.float32)
    
    def _load_video_frames(self, sample):
        """Load video frames for a sample"""
        try:
            cap = cv2.VideoCapture(sample['video_path'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample['start_frame'])
            
            frames = []
            for i in range(self.window_size):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                if self.transform:
                    frame = self.transform(frame)
                
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == self.window_size:
                return torch.stack(frames)
            else:
                return None
                
        except Exception as e:
            print(f"Error loading video frames: {e}")
            return None
