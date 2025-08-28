# 📊 Comprehensive Data Handling Guide

*Complete guide for downloading, loading, processing, and integrating all datasets in the VitalLens multi-modal pipeline*

## 📋 Table of Contents

1. [Overview](#overview)
2. [Dataset Registry](#dataset-registry)
3. [Quick Start](#quick-start)
4. [Download Automation](#download-automation)
5. [Data Loading](#data-loading)
6. [Preprocessing Pipelines](#preprocessing-pipelines)
7. [Multi-Modal Integration](#multi-modal-integration)
8. [Quality Assessment](#quality-assessment)
9. [Dataset-Specific Guides](#dataset-specific-guides)
10. [Troubleshooting](#troubleshooting)
11. [Individual Dataset Examination](#individual-dataset-examination)

## Overview

The VitalLens pipeline supports 22 datasets across multiple modalities (rPPG, emotion, eye-tracking, audio) with automated download, preprocessing, and integration capabilities. This guide provides complete instructions for handling all supported datasets.

### Supported Dataset Categories
- **rPPG Datasets**: 3 datasets (3,600+ recordings)
- **Emotion Datasets**: 7 datasets (478K+ samples)
- **Eye-Tracking Datasets**: 4 datasets (behavioral analysis)
- **Multi-Modal Datasets**: 5 datasets (cross-modal training)
- **Audio Emotion**: 3 datasets (speech emotion recognition)

## Dataset Registry

All datasets are cataloged in our comprehensive registry:

```bash
# View complete dataset registry
cat rich_datasets/dataset_registry.json

# View dataset summary report
cat rich_datasets/rich_dataset_report.md
```

### High-Priority Datasets (Immediate Download)
1. **boltuix/emotions-dataset** - 155K emotion samples
2. **ChristophSchuhmann/emotions** - 212K image-text pairs
3. **AdamCodd/yolo-emotions** - 155K with bounding boxes
4. **kyegorov/mcd_rppg** - 3,600 rPPG recordings
5. **julienmercier/eyetracking** - Eye-tracking behavioral data

## Quick Start

### Complete Dataset Setup (One Command)
```bash
# Download all high-priority datasets
python scripts/create_rich_dataset.py --download-all --priority high

# Download specific dataset categories
python scripts/create_rich_dataset.py --category emotion --download
python scripts/create_rich_dataset.py --category rppg --download
python scripts/create_rich_dataset.py --category eyetracking --download

# Download individual datasets
python scripts/create_rich_dataset.py --dataset "boltuix/emotions-dataset" --download
```

### Verify Dataset Installation
```bash
# Check downloaded datasets
python scripts/create_rich_dataset.py --verify-all

# Test data loading
python test_pipeline.py --test-data-loading
```

## Download Automation

### HuggingFace Datasets
The pipeline automatically handles HuggingFace dataset downloads using the `datasets` library:

```python
from datasets import load_dataset
from scripts.create_rich_dataset import RichDatasetCollector

# Initialize collector
collector = RichDatasetCollector()

# Download specific HuggingFace dataset
dataset = collector.download_huggingface_dataset(
    dataset_name="boltuix/emotions-dataset",
    cache_dir="./data/huggingface_cache"
)

# Download with specific configuration
dataset = collector.download_huggingface_dataset(
    dataset_name="ChristophSchuhmann/emotions",
    split="train",
    streaming=False,  # Download full dataset
    cache_dir="./data/huggingface_cache"
)
```

### Kaggle Datasets
For Kaggle datasets, the pipeline uses the Kaggle API:

```python
# Use the consolidated download script for all datasets
from scripts.download_datasets import download_fer2013, download_hf_emotion_datasets

# Download emotion datasets
download_fer2013()
download_hf_emotion_datasets()

# For traditional rPPG datasets with Google Drive integration
from scripts.download_datasets import download_traditional_rppg

# Download UBFC-rPPG dataset
download_traditional_rppg("UBFC-rPPG")

# Download specific files only
downloader.download_dataset(
    dataset_id="mh0386/facial-emotions-detection",
    download_path="./data/kaggle_datasets/",
    files=["train.csv", "test.csv"]  # Optional: specific files
)
```

### Traditional rPPG Datasets
For established rPPG datasets (UBFC-rPPG, PURE, COHFACE):

```python
from scripts.download_datasets import download_huggingface_datasets

# Download all HuggingFace datasets
download_huggingface_datasets()

# Download UBFC-rPPG dataset
downloader.download_ubfc_rppg(
    download_path="./data/rppg_datasets/",
    extract=True
)

# Download PURE dataset
downloader.download_pure_dataset(
    download_path="./data/rppg_datasets/",
    subjects="all"  # or specific subject list
)

# Download COHFACE dataset
downloader.download_cohface(
    download_path="./data/rppg_datasets/"
)
```

## Data Loading

### Multi-Modal Dataset Loader
The `RPPGEmotionDataset` class handles all dataset types:

```python
from src.data.dataset import RPPGEmotionDataset
import torch
from torch.utils.data import DataLoader

# Initialize multi-modal dataset
dataset = RPPGEmotionDataset(
    data_dir="./data/",
    sequence_length=150,  # 5 seconds at 30 FPS
    include_audio=True,
    include_eyetracking=True,
    emotion_classes=7,  # angry, disgust, fear, happy, neutral, sad, surprise
    transform=None  # Will use default transforms
)

# Create data loader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Iterate through data
for batch in dataloader:
    video_frames = batch['video']  # (B, T, C, H, W)
    audio_features = batch['audio']  # (B, 1, 128, 128)
    eyetrack_coords = batch['eyetrack']  # (B, 2)
    emotion_labels = batch['emotion']  # (B,)
    rppg_targets = batch['rppg']  # (B, T)
    break
```

### Dataset-Specific Loaders

#### Emotion Dataset Loading
```python
from datasets import load_dataset

# Load emotion datasets
emotions_dataset = load_dataset("boltuix/emotions-dataset")
yolo_emotions = load_dataset("AdamCodd/yolo-emotions")

# Process emotion data
def process_emotion_sample(sample):
    image = sample['image']
    emotion_label = sample['emotion']  # 0-6 for 7 emotion classes
    
    # Apply transforms
    if transform:
        image = transform(image)
    
    return {
        'image': image,
        'emotion': emotion_label
    }
```

#### rPPG Dataset Loading
```python
from src.data.dataset import RPPGEmotionDataset

# Initialize dataset processor
dataset = RPPGEmotionDataset(data_dir='./datasets')

# Load rPPG dataset
rppg_data = processor.load_dataset(
    dataset_path="./data/rppg_datasets/UBFC-rPPG/",
    dataset_type="ubfc",
    subjects=None  # Load all subjects
)

# Process rPPG sample
def process_rppg_sample(video_path, ground_truth_path):
    # Load video frames
    frames = processor.load_video_frames(
        video_path,
        target_fps=30,
        max_frames=150
    )
    
    # Load ground truth signals
    gt_signals = processor.load_ground_truth(
        ground_truth_path,
        signal_types=['ppg', 'hr', 'rr']
    )
    
    return {
        'frames': frames,
        'ppg_signal': gt_signals['ppg'],
        'heart_rate': gt_signals['hr'],
        'resp_rate': gt_signals['rr']
    }
```

#### Eye-Tracking Dataset Loading
```python
# Load eye-tracking datasets
eyetrack_dataset = load_dataset("julienmercier/eyetracking")

def process_eyetrack_sample(sample):
    gaze_x = sample['gaze_x']
    gaze_y = sample['gaze_y']
    timestamp = sample['timestamp']
    
    # Normalize coordinates to [0, 1]
    normalized_coords = torch.tensor([
        gaze_x / sample['screen_width'],
        gaze_y / sample['screen_height']
    ])
    
    return {
        'gaze_coords': normalized_coords,
        'timestamp': timestamp
    }
```

## Preprocessing Pipelines

### Video Preprocessing
```python
from torchvision import transforms
import cv2

# Standard video preprocessing pipeline
video_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# Advanced preprocessing with face detection
from src.data.dataset import RPPGEmotionDataset

# Face detection and ROI extraction handled by dataset class
dataset = RPPGEmotionDataset(data_dir='./datasets', enable_face_detection=True)

def preprocess_video_advanced(video_frames):
    processed_frames = []
    
    for frame in video_frames:
        # Detect face
        face_bbox = face_detector.detect_face(frame)
        
        if face_bbox is not None:
            # Extract ROI
            roi_frame = roi_extractor.extract_roi(frame, face_bbox)
            
            # Apply transforms
            processed_frame = video_transform(roi_frame)
            processed_frames.append(processed_frame)
    
    return torch.stack(processed_frames)
```

### Audio Preprocessing
```python
import librosa
import numpy as np

def preprocess_audio(audio_path, target_sr=16000, n_mels=128):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=512,
        win_length=2048
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    normalized_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std()
    
    return torch.tensor(normalized_spec).unsqueeze(0)  # Add channel dimension
```

### Data Augmentation
```python
# Comprehensive augmentation pipeline
augmentation_transform = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.RandomResizedCrop(
        size=(224, 224),
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## Multi-Modal Integration

### Synchronized Multi-Modal Loading
```python
class MultiModalSynchronizer:
    def __init__(self, video_fps=30, audio_sr=16000):
        self.video_fps = video_fps
        self.audio_sr = audio_sr
    
    def synchronize_modalities(self, video_frames, audio_data, eyetrack_data):
        # Calculate time alignment
        video_duration = len(video_frames) / self.video_fps
        audio_duration = len(audio_data) / self.audio_sr
        
        # Trim to shortest duration
        min_duration = min(video_duration, audio_duration)
        
        # Synchronize video
        target_frames = int(min_duration * self.video_fps)
        sync_video = video_frames[:target_frames]
        
        # Synchronize audio
        target_audio_samples = int(min_duration * self.audio_sr)
        sync_audio = audio_data[:target_audio_samples]
        
        # Interpolate eye-tracking data
        sync_eyetrack = self.interpolate_eyetrack(
            eyetrack_data, 
            target_duration=min_duration
        )
        
        return sync_video, sync_audio, sync_eyetrack
```

### Cross-Modal Validation
```python
def validate_multimodal_sample(sample):
    """Validate that all modalities are properly aligned"""
    checks = {
        'video_shape': sample['video'].shape[0] == 150,  # 5 seconds at 30 FPS
        'audio_shape': sample['audio'].shape == (1, 128, 128),  # Mel-spectrogram
        'eyetrack_shape': sample['eyetrack'].shape == (2,),  # x, y coordinates
        'emotion_valid': 0 <= sample['emotion'] <= 6,  # 7 emotion classes
        'rppg_length': len(sample['rppg']) == 150  # Same as video frames
    }
    
    return all(checks.values()), checks
```

## Quality Assessment

### Automatic Quality Filtering
```python
from src.data.dataset import RPPGEmotionDataset

# Quality assessment integrated into dataset class
dataset = RPPGEmotionDataset(data_dir='./datasets', quality_threshold=0.7)

def assess_sample_quality(video_frames, rppg_signal=None):
    quality_metrics = {}
    
    # Video quality assessment
    quality_metrics['brightness'] = quality_assessor.assess_brightness(video_frames)
    quality_metrics['motion'] = quality_assessor.assess_motion(video_frames)
    quality_metrics['face_visibility'] = quality_assessor.assess_face_visibility(video_frames)
    
    # rPPG signal quality (if available)
    if rppg_signal is not None:
        quality_metrics['snr'] = quality_assessor.calculate_snr(rppg_signal)
        quality_metrics['signal_stability'] = quality_assessor.assess_stability(rppg_signal)
    
    # Overall quality score
    quality_metrics['overall_score'] = quality_assessor.compute_overall_score(quality_metrics)
    
    return quality_metrics

# Filter high-quality samples
def filter_high_quality_samples(dataset, min_quality_score=0.7):
    filtered_samples = []
    
    for sample in dataset:
        quality = assess_sample_quality(sample['video'], sample.get('rppg'))
        
        if quality['overall_score'] >= min_quality_score:
            sample['quality_metrics'] = quality
            filtered_samples.append(sample)
    
    return filtered_samples
```

### SNR-Based Filtering
```python
def calculate_snr(signal, fs=30, hr_range=(0.7, 4.0)):
    """Calculate Signal-to-Noise Ratio for rPPG signals"""
    from scipy import signal as scipy_signal
    
    # Apply bandpass filter for heart rate range
    nyquist = fs / 2
    low = hr_range[0] / nyquist
    high = hr_range[1] / nyquist
    
    b, a = scipy_signal.butter(4, [low, high], btype='band')
    filtered_signal = scipy_signal.filtfilt(b, a, signal)
    
    # Calculate SNR
    signal_power = np.mean(filtered_signal ** 2)
    noise_power = np.mean((signal - filtered_signal) ** 2)
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db
```

## Dataset-Specific Guides

For detailed dataset information, see:
- [Consolidated Dataset Registry](RESOURCES.json) - Comprehensive dataset information with metadata, loading patterns, and code examples for all 22 datasets
- [Individual Dataset Guides](INDIVIDUAL_DATASET_GUIDES.md) - Streamlined guide referencing the consolidated dataset information

### 1. boltuix/emotions-dataset
```python
# Download and process
dataset = load_dataset("boltuix/emotions-dataset")

# Structure: {'image': PIL.Image, 'emotion': int}
# Emotions: 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise

for sample in dataset['train']:
    image = sample['image']
    emotion = sample['emotion']
    
    # Apply preprocessing
    processed_image = video_transform(image)
```

### 2. kyegorov/mcd_rppg
```python
# MCD rPPG dataset with real physiological data
dataset = load_dataset("kyegorov/mcd_rppg")

# Structure: {'video': video_path, 'ppg': signal_array, 'hr': float, 'subject_id': str}

for sample in dataset['train']:
    video_path = sample['video']
    ppg_signal = sample['ppg']
    heart_rate = sample['hr']
    
    # Load and process video
    frames = load_video_frames(video_path)
    processed_frames = preprocess_video_advanced(frames)
```

### 3. AdamCodd/yolo-emotions
```python
# YOLO emotions with bounding boxes
dataset = load_dataset("AdamCodd/yolo-emotions")

# Structure: {'image': PIL.Image, 'emotion': int, 'bbox': [x, y, w, h]}

for sample in dataset['train']:
    image = sample['image']
    emotion = sample['emotion']
    bbox = sample['bbox']  # Face bounding box
    
    # Crop face region
    x, y, w, h = bbox
    face_crop = image.crop((x, y, x+w, y+h))
    
    # Process cropped face
    processed_face = video_transform(face_crop)
```

### 4. julienmercier/eyetracking
```python
# Eye-tracking behavioral data
dataset = load_dataset("julienmercier/eyetracking")

# Structure: {'gaze_x': float, 'gaze_y': float, 'timestamp': float, 'screen_width': int, 'screen_height': int}

for sample in dataset['train']:
    gaze_coords = normalize_gaze_coordinates(
        sample['gaze_x'], 
        sample['gaze_y'],
        sample['screen_width'],
        sample['screen_height']
    )
```

## Batch Processing

### Efficient Batch Loading
```python
def create_efficient_dataloader(dataset_config):
    """Create optimized dataloader for large-scale training"""
    
    dataset = RPPGEmotionDataset(**dataset_config)
    
    # Optimized dataloader settings
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Adjust based on GPU memory
        shuffle=True,
        num_workers=8,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2,  # Prefetch batches
        drop_last=True  # Consistent batch sizes
    )
    
    return dataloader
```

### Memory-Efficient Processing
```python
def process_large_dataset_streaming(dataset_name, batch_size=32):
    """Process large datasets without loading everything into memory"""
    
    # Use streaming for large datasets
    dataset = load_dataset(dataset_name, streaming=True)
    
    batch = []
    for sample in dataset['train']:
        processed_sample = preprocess_sample(sample)
        batch.append(processed_sample)
        
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []
    
    # Process remaining samples
    if batch:
        yield torch.stack(batch)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. HuggingFace Dataset Download Failures
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/datasets/
python scripts/create_rich_dataset.py --dataset "boltuix/emotions-dataset" --download --force

# Use streaming for large datasets
python -c "
from datasets import load_dataset
dataset = load_dataset('ChristophSchuhmann/emotions', streaming=True)
print('Streaming dataset loaded successfully')
"
```

#### 2. Kaggle API Authentication
```bash
# Setup Kaggle credentials
mkdir -p ~/.kaggle
echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Test authentication
kaggle datasets list --search emotion
```

#### 3. Memory Issues with Large Datasets
```python
# Use memory mapping for large arrays
import numpy as np

# Save processed data as memory-mapped arrays
def save_as_memmap(data, filepath):
    memmap_array = np.memmap(
        filepath, 
        dtype=np.float32, 
        mode='w+', 
        shape=data.shape
    )
    memmap_array[:] = data[:]
    del memmap_array  # Flush to disk

# Load memory-mapped data
def load_memmap(filepath, shape, dtype=np.float32):
    return np.memmap(filepath, dtype=dtype, mode='r', shape=shape)
```

#### 4. Video Loading Issues
```python
# Robust video loading with fallbacks
def robust_video_load(video_path):
    try:
        # Try OpenCV first
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    except:
        try:
            # Fallback to imageio
            import imageio
            reader = imageio.get_reader(video_path)
            frames = [frame for frame in reader]
            reader.close()
            return frames
        except:
            print(f"Failed to load video: {video_path}")
            return None
```

#### 5. Audio Processing Issues
```python
# Handle various audio formats
def robust_audio_load(audio_path, target_sr=16000):
    try:
        # Try librosa first
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio, sr
    except:
        try:
            # Fallback to soundfile
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            return audio, target_sr
        except:
            print(f"Failed to load audio: {audio_path}")
            return None, None
```

## Performance Optimization

### Dataset Caching
```python
# Cache preprocessed data for faster loading
import pickle
import os

class DatasetCache:
    def __init__(self, cache_dir="./data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, dataset_name, sample_id):
        return os.path.join(self.cache_dir, f"{dataset_name}_{sample_id}.pkl")
    
    def cache_sample(self, dataset_name, sample_id, processed_sample):
        cache_path = self.get_cache_path(dataset_name, sample_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(processed_sample, f)
    
    def load_cached_sample(self, dataset_name, sample_id):
        cache_path = self.get_cache_path(dataset_name, sample_id)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
```

### Parallel Processing
```python
from multiprocessing import Pool
import functools

def parallel_preprocess_dataset(dataset, num_workers=8):
    """Preprocess dataset samples in parallel"""
    
    with Pool(num_workers) as pool:
        processed_samples = pool.map(
            preprocess_sample,
            dataset
        )
    
    return processed_samples

# Partial function for complex preprocessing
def preprocess_with_config(sample, config):
    return preprocess_sample(sample, **config)

def parallel_preprocess_with_config(dataset, config, num_workers=8):
    preprocess_func = functools.partial(preprocess_with_config, config=config)
    
    with Pool(num_workers) as pool:
        processed_samples = pool.map(preprocess_func, dataset)
    
    return processed_samples
```

## Validation and Testing

### Dataset Integrity Checks
```python
def validate_dataset_integrity(dataset_path):
    """Comprehensive dataset validation"""
    
    validation_results = {
        'total_samples': 0,
        'valid_samples': 0,
        'corrupted_files': [],
        'missing_labels': [],
        'quality_distribution': {}
    }
    
    for sample_id, sample in enumerate(dataset):
        validation_results['total_samples'] += 1
        
        try:
            # Check file integrity
            if 'video' in sample:
                frames = load_video_frames(sample['video'])
                if frames is None or len(frames) == 0:
                    validation_results['corrupted_files'].append(sample_id)
                    continue
            
            # Check labels
            if 'emotion' in sample:
                if not (0 <= sample['emotion'] <= 6):
                    validation_results['missing_labels'].append(sample_id)
                    continue
            
            # Assess quality
            if 'video' in sample:
                quality = assess_sample_quality(frames)
                quality_level = 'high' if quality['overall_score'] > 0.7 else 'low'
                validation_results['quality_distribution'][quality_level] = \
                    validation_results['quality_distribution'].get(quality_level, 0) + 1
            
            validation_results['valid_samples'] += 1
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            validation_results['corrupted_files'].append(sample_id)
    
    return validation_results
```

### Performance Benchmarking
```python
import time

def benchmark_data_loading(dataloader, num_batches=100):
    """Benchmark data loading performance"""
    
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Simulate processing time
        _ = batch['video'].shape
        
        if i % 10 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (i + 1) * dataloader.batch_size / elapsed
            print(f"Batch {i+1}/{num_batches}, {samples_per_sec:.1f} samples/sec")
    
    total_time = time.time() - start_time
    total_samples = num_batches * dataloader.batch_size
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Average throughput: {total_samples/total_time:.1f} samples/sec")
    
    return total_samples / total_time
```

## Summary

This comprehensive data handling guide covers all aspects of working with the 22 datasets in the VitalLens pipeline:

- **Automated Download**: Scripts for HuggingFace, Kaggle, and traditional rPPG datasets
- **Multi-Modal Loading**: Unified interface for video, audio, and eye-tracking data
- **Quality Assessment**: Automatic filtering and SNR-based validation
- **Performance Optimization**: Caching, parallel processing, and memory efficiency
- **Troubleshooting**: Solutions for common issues and robust error handling

The pipeline is designed to handle the complete data workflow from raw dataset download to training-ready batches, with comprehensive quality control and performance optimization throughout.

## Individual Dataset Examination

For detailed examination of each of the 22 datasets with specific loading patterns, preprocessing requirements, and integration procedures, see:

**[Individual Dataset Guides](INDIVIDUAL_DATASET_GUIDES.md)**

This comprehensive guide covers:
- **7 HuggingFace datasets** with specific loading patterns
- **15 Kaggle resources** for technique extraction
- **Traditional rPPG datasets** with existing infrastructure
- **Multi-modal synchronization** procedures
- **Quality assessment** and validation frameworks

Each dataset has been individually examined with practical code examples, preprocessing steps, and integration patterns suitable for the VitalLens multi-modal architecture.

---

*Last updated: August 28, 2025*  
*Covers: 22 datasets, 5 modalities, complete automation pipeline*
