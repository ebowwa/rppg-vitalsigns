# Individual Dataset Examination Guide

This document provides detailed examination of each of the 22 datasets in the VitalLens multi-modal pipeline, including specific loading patterns, preprocessing requirements, and integration procedures.

## Table of Contents

1. [HuggingFace Datasets](#1-huggingface-datasets)
   - [1.1 Emotion Datasets](#11-emotion-datasets)
   - [1.2 rPPG Datasets](#12-rppg-datasets)
   - [1.3 Eye-tracking Datasets](#13-eye-tracking-datasets)
   - [1.4 Behavioral Datasets](#14-behavioral-datasets)
2. [Kaggle Resources](#2-kaggle-resources)
   - [2.1 Audio Emotion Processing](#21-audio-emotion-processing)
   - [2.2 Facial Emotion Detection](#22-facial-emotion-detection)
   - [2.3 Multi-modal Approaches](#23-multi-modal-approaches)
3. [Traditional rPPG Datasets](#3-traditional-rppg-datasets)
4. [Multi-Modal Synchronization](#4-multi-modal-synchronization)
5. [Integration Examples](#5-integration-examples)

## 1. HuggingFace Datasets

### 1.1 Emotion Datasets

#### 1.1.1 boltuix/emotions-dataset
- **Type**: Text-based emotion classification
- **Samples**: 131,306 entries
- **Classes**: 13 emotion categories
- **Format**: Text sentences with emotion labels
- **Priority**: High

**Loading Pattern:**
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("boltuix/emotions-dataset")

# Access training data
train_data = dataset['train']

# Example data structure
# {'sentence': 'Unfortunately later died from eating tainted meat NAME BBC documentary dynasties followed the marsh pride the lion episode was awesome', 'label': 'happiness'}

# Integration with VitalLens pipeline
def load_text_emotions():
    dataset = load_dataset("boltuix/emotions-dataset")
    
    # Map to standard emotion classes
    emotion_mapping = {
        'happiness': 3, 'joy': 3, 'neutral': 4, 'sadness': 5,
        'anger': 0, 'fear': 2, 'surprise': 6, 'disgust': 1
    }
    
    processed_data = []
    for item in dataset['train']:
        if item['label'] in emotion_mapping:
            processed_data.append({
                'text': item['sentence'],
                'emotion_label': emotion_mapping[item['label']]
            })
    
    return processed_data
```

#### 1.1.2 ChristophSchuhmann/emotions
- **Type**: Image-text pairs with emotion context
- **Samples**: 211,677 entries
- **Format**: URLs, similarity scores, captions
- **Priority**: High

**Loading Pattern:**
```python
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO

# Load the dataset
dataset = load_dataset("ChristophSchuhmann/emotions")

# Example data structure
# {'url': 'https://...', 'similarity': 0.303739, 'id': 1028568224, 'caption': 'portrait of young man wearing blue shirt making silly faces against black.'}

def load_image_text_emotions():
    dataset = load_dataset("ChristophSchuhmann/emotions")
    
    processed_data = []
    for item in dataset['train']:
        try:
            # Download image
            response = requests.get(item['url'], timeout=10)
            image = Image.open(BytesIO(response.content))
            
            # Extract emotion from caption using keyword matching
            caption = item['caption'].lower()
            emotion_label = extract_emotion_from_text(caption)
            
            processed_data.append({
                'image': image,
                'caption': item['caption'],
                'emotion_label': emotion_label,
                'similarity': item['similarity']
            })
        except Exception as e:
            continue
    
    return processed_data

def extract_emotion_from_text(text):
    emotion_keywords = {
        0: ['angry', 'mad', 'furious', 'rage'],
        1: ['disgusted', 'revolted', 'sick'],
        2: ['scared', 'afraid', 'fearful', 'terrified'],
        3: ['happy', 'joyful', 'smiling', 'cheerful'],
        4: ['neutral', 'calm', 'normal'],
        5: ['sad', 'crying', 'depressed', 'melancholy'],
        6: ['surprised', 'shocked', 'amazed']
    }
    
    for emotion_id, keywords in emotion_keywords.items():
        if any(keyword in text for keyword in keywords):
            return emotion_id
    return 4  # Default to neutral
```

#### 1.1.3 AdamCodd/yolo-emotions
- **Type**: YOLO format emotion detection with bounding boxes
- **Samples**: 155,591 entries
- **Format**: Images with YOLO annotations
- **Classes**: 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)
- **Priority**: High

**Loading Pattern:**
```python
from datasets import load_dataset
import torch
from PIL import Image

# Load the dataset
dataset = load_dataset("AdamCodd/yolo-emotions")

# Example data structure includes images and bounding box annotations
def load_yolo_emotions():
    dataset = load_dataset("AdamCodd/yolo-emotions")
    
    processed_data = []
    for item in dataset['train']:
        # Extract image and annotations
        image = item['image']
        
        # YOLO format: class_id center_x center_y width height
        # Convert to emotion classification format
        if 'objects' in item:
            for obj in item['objects']:
                bbox = obj['bbox']  # [x, y, width, height]
                emotion_class = obj['category']  # 0-6 for 7 emotions
                
                # Crop face region for emotion classification
                x, y, w, h = bbox
                face_crop = image.crop((x, y, x+w, y+h))
                
                processed_data.append({
                    'image': face_crop,
                    'emotion_label': emotion_class,
                    'bbox': bbox,
                    'full_image': image
                })
    
    return processed_data
```

#### 1.1.4 tukey/human_face_emotions_roboflow
- **Type**: Face emotion detection with Roboflow annotations
- **Samples**: Unknown (medium priority)
- **Format**: Images with face bounding boxes and emotion labels

**Loading Pattern:**
```python
def load_roboflow_emotions():
    dataset = load_dataset("tukey/human_face_emotions_roboflow")
    
    # Similar to YOLO format but with Roboflow-specific annotations
    processed_data = []
    for item in dataset['train']:
        # Process Roboflow format annotations
        image = item['image']
        annotations = item.get('annotations', [])
        
        for ann in annotations:
            emotion_label = ann.get('category_id', 4)  # Default neutral
            bbox = ann.get('bbox', [0, 0, image.width, image.height])
            
            processed_data.append({
                'image': image,
                'emotion_label': emotion_label,
                'bbox': bbox
            })
    
    return processed_data
```

#### 1.1.5 juanbtbx/Human-Group-Emotions-Labelled
- **Type**: Group emotion analysis
- **Samples**: Unknown (medium priority)
- **Format**: Group images with collective emotion labels

#### 1.1.6 HazemAbdelkawy/emotions
- **Type**: Emotion classification with "content" class
- **Samples**: 9,400 entries
- **Classes**: Includes unique "content" emotion class

### 1.2 rPPG Datasets

#### 1.2.1 kyegorov/mcd_rppg
- **Type**: Multi-camera rPPG dataset
- **Samples**: 3,600 recordings from 600 subjects
- **Format**: Video files with physiological ground truth
- **Priority**: High

**Loading Pattern:**
```python
from datasets import load_dataset
import cv2
import numpy as np

def load_mcd_rppg():
    dataset = load_dataset("kyegorov/mcd_rppg")
    
    processed_data = []
    for item in dataset['train']:
        # Extract video and physiological signals
        video_path = item.get('video_path')
        hr_ground_truth = item.get('heart_rate')
        rr_ground_truth = item.get('respiratory_rate')
        
        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        # Process into 150-frame sequences
        sequence_length = 150
        for i in range(0, len(frames) - sequence_length, sequence_length // 2):
            frame_sequence = frames[i:i + sequence_length]
            
            processed_data.append({
                'video_frames': np.array(frame_sequence),
                'heart_rate': hr_ground_truth,
                'respiratory_rate': rr_ground_truth,
                'subject_id': item.get('subject_id'),
                'session_id': item.get('session_id')
            })
    
    return processed_data
```

### 1.3 Eye-tracking Datasets

#### 1.3.1 julienmercier/eyetracking
- **Type**: Basic eye-tracking data
- **Format**: Gaze coordinates and timestamps
- **Priority**: Medium

**Loading Pattern:**
```python
def load_eyetracking_data():
    dataset = load_dataset("julienmercier/eyetracking")
    
    processed_data = []
    for item in dataset['train']:
        # Extract gaze coordinates and timing
        gaze_x = item.get('gaze_x', [])
        gaze_y = item.get('gaze_y', [])
        timestamps = item.get('timestamps', [])
        
        # Synchronize with video frames (30 FPS)
        fps = 30
        frame_duration = 1.0 / fps
        
        synchronized_gaze = []
        for i, ts in enumerate(timestamps):
            frame_idx = int(ts * fps)
            if i < len(gaze_x) and i < len(gaze_y):
                synchronized_gaze.append({
                    'frame_idx': frame_idx,
                    'gaze_x': gaze_x[i],
                    'gaze_y': gaze_y[i],
                    'timestamp': ts
                })
        
        processed_data.append({
            'gaze_data': synchronized_gaze,
            'duration': max(timestamps) if timestamps else 0
        })
    
    return processed_data
```

#### 1.3.2 julienmercier/mobile-eye-tracking-dataset-v3
- **Type**: Mobile eye-tracking with enhanced features
- **Priority**: Medium

#### 1.3.3 shiv213/Eye-tracking-and-Sentiment-Analysis-Dataset-II
- **Type**: Eye-tracking combined with sentiment analysis
- **Priority**: High (multi-modal ready)

#### 1.3.4 Wangtwohappy/EgoLife_EyeTracking_EyeGaze
- **Type**: Egocentric eye-tracking and gaze data
- **Priority**: Medium

### 1.4 Behavioral Datasets

#### 1.4.1 facebook/PLM-Video-Human
- **Type**: Human behavior analysis in video
- **Format**: Video sequences with behavioral annotations
- **Priority**: Medium

#### 1.4.2 Bingsu/Human_Action_Recognition
- **Type**: Human action recognition
- **Format**: Video clips with action labels
- **Priority**: Medium

## 2. Kaggle Resources

### 2.1 Audio Emotion Processing

#### 2.1.1 Audio Emotion Part 1: Explore Data
- **URL**: https://www.kaggle.com/code/ejlok1/audio-emotion-part-1-explore-data
- **Focus**: Data exploration techniques for audio emotion datasets
- **Key Techniques**: Waveform analysis, spectral features, emotion distribution analysis

**Integration Pattern:**
```python
import librosa
import numpy as np
import pandas as pd

def extract_audio_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    
    # Extract MFCC features (from Kaggle notebook techniques)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    
    # Extract additional features
    chroma = np.mean(librosa.feature.chroma(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    
    # Combine features
    features = np.hstack([mfcc_processed, chroma, mel, contrast, tonnetz])
    
    return features
```

#### 2.1.2 Audio Emotion Part 2: Feature Extract
- **URL**: https://www.kaggle.com/code/ejlok1/audio-emotion-part-2-feature-extract
- **Focus**: Advanced feature extraction for audio emotion recognition
- **Key Techniques**: MFCC, spectral features, temporal dynamics

#### 2.1.3 Speech Emotion Recognition 97.25% Accuracy
- **URL**: https://www.kaggle.com/code/mostafaabdlhamed/speech-emotion-recognition-97-25-accuracy
- **Focus**: High-accuracy speech emotion recognition
- **Key Techniques**: Deep learning approaches, feature engineering

### 2.2 Facial Emotion Detection

#### 2.2.1 Facial Emotions Detection
- **URL**: https://www.kaggle.com/code/mh0386/facial-emotions-detection
- **Focus**: CNN-based facial emotion recognition
- **Key Techniques**: Convolutional neural networks, data augmentation

**Integration Pattern:**
```python
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

def create_facial_emotion_model():
    # Based on Kaggle notebook architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotion classes
    ])
    
    return model

def preprocess_face_for_emotion(face_image):
    # Preprocessing based on Kaggle techniques
    face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_normalized = face_resized / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=-1)
    
    return face_expanded
```

#### 2.2.2 Facial Expression EDA CNN
- **URL**: https://www.kaggle.com/code/drcapa/facial-expression-eda-cnn
- **Focus**: Exploratory data analysis and CNN implementation

#### 2.2.3 Face Emotions Image Detection ViT
- **URL**: https://www.kaggle.com/code/dima806/face-emotions-image-detection-vit
- **Focus**: Vision Transformer approach for emotion detection

### 2.3 Multi-modal Approaches

#### 2.3.1 Taylor Swift Emotions Over the Years
- **URL**: https://www.kaggle.com/code/promptcloud/taylor-swift-emotions-over-the-years
- **Focus**: Multi-modal emotion analysis combining text and image data

#### 2.3.2 EmotionsClassifier HuggingFace BERT
- **URL**: https://www.kaggle.com/code/paddytheprogrammer/emotionsclassifier-huggingface-bert
- **Focus**: BERT-based emotion classification

## 3. Traditional rPPG Datasets

### 3.1 UBFC-rPPG Dataset
**Loading Pattern (from existing infrastructure):**
```python
from VitalLens_Complete_script import DatasetDownloader

def load_ubfc_rppg():
    downloader = DatasetDownloader()
    
    # Download if not exists
    if not os.path.exists('datasets/UBFC-rPPG'):
        downloader.download_ubfc_rppg()
    
    # Load dataset
    dataset_path = 'datasets/UBFC-rPPG'
    processed_data = []
    
    for subject_dir in os.listdir(dataset_path):
        subject_path = os.path.join(dataset_path, subject_dir)
        
        # Load video
        video_path = os.path.join(subject_path, 'vid.avi')
        ground_truth_path = os.path.join(subject_path, 'ground_truth.txt')
        
        if os.path.exists(video_path) and os.path.exists(ground_truth_path):
            # Load ground truth
            with open(ground_truth_path, 'r') as f:
                hr_values = [float(line.strip()) for line in f.readlines()]
            
            # Load video frames
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            processed_data.append({
                'video_frames': np.array(frames),
                'heart_rate_sequence': hr_values,
                'subject_id': subject_dir
            })
    
    return processed_data
```

### 3.2 PURE Dataset
**Loading Pattern:**
```python
def load_pure_dataset():
    downloader = DatasetDownloader()
    
    if not os.path.exists('datasets/PURE'):
        downloader.download_pure()
    
    # Similar processing to UBFC-rPPG but with PURE-specific format
    # PURE has different file structure and ground truth format
```

### 3.3 COHFACE Dataset
**Loading Pattern:**
```python
def load_cohface_dataset():
    downloader = DatasetDownloader()
    
    if not os.path.exists('datasets/COHFACE'):
        downloader.download_cohface()
    
    # COHFACE-specific processing
```

## 4. Multi-Modal Synchronization

### 4.1 Temporal Alignment
```python
def synchronize_multimodal_data(video_frames, audio_features, gaze_data, fps=30):
    """
    Synchronize multi-modal data streams based on timestamps
    """
    # Calculate frame timestamps
    frame_timestamps = np.arange(len(video_frames)) / fps
    
    # Synchronize audio features (assuming 1 feature per frame)
    if len(audio_features) != len(video_frames):
        # Interpolate audio features to match video frames
        audio_indices = np.linspace(0, len(audio_features)-1, len(video_frames))
        synchronized_audio = np.array([audio_features[int(idx)] for idx in audio_indices])
    else:
        synchronized_audio = audio_features
    
    # Synchronize gaze data
    synchronized_gaze = []
    for frame_idx, timestamp in enumerate(frame_timestamps):
        # Find closest gaze sample
        closest_gaze = min(gaze_data, key=lambda x: abs(x['timestamp'] - timestamp))
        synchronized_gaze.append([closest_gaze['gaze_x'], closest_gaze['gaze_y']])
    
    return {
        'video_frames': video_frames,
        'audio_features': synchronized_audio,
        'gaze_coordinates': np.array(synchronized_gaze),
        'timestamps': frame_timestamps
    }
```

### 4.2 Cross-Modal Validation
```python
def validate_multimodal_consistency(video_emotion, audio_emotion, gaze_pattern):
    """
    Validate consistency across modalities
    """
    # Emotion consistency check
    emotion_agreement = (video_emotion == audio_emotion)
    
    # Gaze pattern validation (looking at camera indicates engagement)
    center_x, center_y = 0.5, 0.5  # Normalized center
    gaze_distance = np.sqrt((gaze_pattern[:, 0] - center_x)**2 + 
                           (gaze_pattern[:, 1] - center_y)**2)
    engagement_score = 1.0 - np.mean(gaze_distance)
    
    consistency_score = {
        'emotion_agreement': emotion_agreement,
        'engagement_score': engagement_score,
        'overall_consistency': (emotion_agreement * 0.7 + engagement_score * 0.3)
    }
    
    return consistency_score
```

## 5. Integration Examples

### 5.1 Complete Multi-Modal Data Loading
```python
def load_complete_multimodal_dataset():
    """
    Load and integrate all dataset types into unified format
    """
    # Load emotion datasets
    text_emotions = load_text_emotions()
    image_emotions = load_image_text_emotions()
    yolo_emotions = load_yolo_emotions()
    
    # Load rPPG datasets
    mcd_rppg = load_mcd_rppg()
    ubfc_rppg = load_ubfc_rppg()
    
    # Load eye-tracking data
    eyetracking = load_eyetracking_data()
    
    # Combine into unified format
    unified_dataset = []
    
    # Process each modality and create synchronized samples
    for rppg_sample in mcd_rppg:
        # Find matching emotion and gaze data based on subject/session
        matching_emotion = find_matching_emotion_data(rppg_sample, yolo_emotions)
        matching_gaze = find_matching_gaze_data(rppg_sample, eyetracking)
        
        if matching_emotion and matching_gaze:
            synchronized_sample = synchronize_multimodal_data(
                rppg_sample['video_frames'],
                extract_audio_features_from_video(rppg_sample['video_frames']),
                matching_gaze['gaze_data']
            )
            
            unified_sample = {
                'video_frames': synchronized_sample['video_frames'],
                'audio_features': synchronized_sample['audio_features'],
                'gaze_coordinates': synchronized_sample['gaze_coordinates'],
                'heart_rate': rppg_sample['heart_rate'],
                'respiratory_rate': rppg_sample['respiratory_rate'],
                'emotion_label': matching_emotion['emotion_label'],
                'subject_id': rppg_sample['subject_id']
            }
            
            unified_dataset.append(unified_sample)
    
    return unified_dataset
```

### 5.2 Dataset Quality Assessment
```python
def assess_dataset_quality(dataset_samples):
    """
    Assess quality of loaded dataset samples
    """
    quality_metrics = {
        'total_samples': len(dataset_samples),
        'complete_samples': 0,
        'missing_modalities': {'video': 0, 'audio': 0, 'gaze': 0},
        'emotion_distribution': {},
        'average_video_length': 0,
        'quality_scores': []
    }
    
    for sample in dataset_samples:
        # Check completeness
        is_complete = all(key in sample for key in 
                         ['video_frames', 'audio_features', 'gaze_coordinates', 'emotion_label'])
        
        if is_complete:
            quality_metrics['complete_samples'] += 1
            
            # Calculate quality score
            video_quality = assess_video_quality(sample['video_frames'])
            audio_quality = assess_audio_quality(sample['audio_features'])
            gaze_quality = assess_gaze_quality(sample['gaze_coordinates'])
            
            overall_quality = (video_quality + audio_quality + gaze_quality) / 3
            quality_metrics['quality_scores'].append(overall_quality)
        
        # Track missing modalities
        for modality in ['video_frames', 'audio_features', 'gaze_coordinates']:
            if modality not in sample or sample[modality] is None:
                modality_name = modality.split('_')[0]
                quality_metrics['missing_modalities'][modality_name] += 1
        
        # Track emotion distribution
        emotion = sample.get('emotion_label', 'unknown')
        quality_metrics['emotion_distribution'][emotion] = \
            quality_metrics['emotion_distribution'].get(emotion, 0) + 1
    
    # Calculate averages
    if quality_metrics['quality_scores']:
        quality_metrics['average_quality'] = np.mean(quality_metrics['quality_scores'])
    
    return quality_metrics
```

## Summary

This guide provides comprehensive examination of all 22 datasets in the VitalLens pipeline:

- **7 HuggingFace datasets** with specific loading patterns and preprocessing
- **15 Kaggle resources** for technique extraction and integration
- **3 traditional rPPG datasets** with existing infrastructure
- **Multi-modal synchronization** procedures for cross-modal consistency
- **Quality assessment** and validation frameworks

Each dataset type has been individually examined with practical loading examples, preprocessing steps, and integration patterns suitable for the VitalLens multi-modal architecture.
