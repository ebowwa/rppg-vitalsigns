#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from PIL import Image
import io

def download_fer2013():
    print("Downloading FER2013 dataset...")
    try:
        dataset = load_dataset("FER2013")
        
        fer2013_dir = Path("./data/fer2013")
        fer2013_dir.mkdir(parents=True, exist_ok=True)
        
        train_data = dataset['train']
        test_data = dataset['test']
        
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        train_df.to_csv(fer2013_dir / "train.csv", index=False)
        test_df.to_csv(fer2013_dir / "test.csv", index=False)
        
        print(f"FER2013 downloaded: {len(train_df)} train, {len(test_df)} test samples")
        
    except Exception as e:
        print(f"Failed to download FER2013: {e}")

def download_hf_emotion_datasets():
    datasets_to_download = [
        'AdamCodd/yolo-emotions',
        'HazemAbdelkawy/emotions', 
        'ChristophSchuhmann/emotions',
        'juanbtbx/Human-Group-Emotions-Labelled'
    ]
    
    data_dir = Path("./data/hf_emotions")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name in datasets_to_download:
        print(f"Downloading {dataset_name}...")
        try:
            dataset = load_dataset(dataset_name, split='train')
            
            safe_name = dataset_name.replace('/', '_')
            dataset_df = pd.DataFrame(dataset)
            dataset_df.to_csv(data_dir / f"{safe_name}.csv", index=False)
            
            print(f"Downloaded {dataset_name}: {len(dataset_df)} samples")
            
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")

def download_mcd_rppg():
    print("Downloading MCD rPPG dataset...")
    try:
        dataset = load_dataset("kyegorov/mcd_rppg", split='train')
        
        mcd_dir = Path("./data/mcd_rppg")
        mcd_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_df = pd.DataFrame(dataset)
        dataset_df.to_csv(mcd_dir / "mcd_rppg.csv", index=False)
        
        print(f"MCD rPPG downloaded: {len(dataset_df)} samples")
        
    except Exception as e:
        print(f"Failed to download MCD rPPG: {e}")

def download_eyetracking_datasets():
    eyetrack_datasets = [
        'julienmercier/eyetracking',
        'julienmercier/mobile-eye-tracking-dataset-v3',
        'shiv213/Eye-tracking-and-Sentiment-Analysis-Dataset-II',
        'Wangtwohappy/EgoLife_EyeTracking_EyeGaze'
    ]
    
    data_dir = Path("./data/eyetracking")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name in eyetrack_datasets:
        print(f"Downloading {dataset_name}...")
        try:
            dataset = load_dataset(dataset_name, split='train')
            
            safe_name = dataset_name.replace('/', '_')
            dataset_df = pd.DataFrame(dataset)
            dataset_df.to_csv(data_dir / f"{safe_name}.csv", index=False)
            
            print(f"Downloaded {dataset_name}: {len(dataset_df)} samples")
            
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")

def create_dataset_summary():
    print("\nCreating dataset summary...")
    
    summary_data = []
    
    fer2013_path = Path("./data/fer2013/train.csv")
    if fer2013_path.exists():
        fer2013_df = pd.read_csv(fer2013_path)
        summary_data.append({
            'dataset': 'FER2013',
            'samples': len(fer2013_df),
            'type': 'emotion_classification',
            'modality': 'image'
        })
    
    hf_emotions_dir = Path("./data/hf_emotions")
    if hf_emotions_dir.exists():
        for csv_file in hf_emotions_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            summary_data.append({
                'dataset': csv_file.stem,
                'samples': len(df),
                'type': 'emotion_classification',
                'modality': 'image'
            })
    
    mcd_path = Path("./data/mcd_rppg/mcd_rppg.csv")
    if mcd_path.exists():
        mcd_df = pd.read_csv(mcd_path)
        summary_data.append({
            'dataset': 'MCD_rPPG',
            'samples': len(mcd_df),
            'type': 'physiological',
            'modality': 'video'
        })
    
    eyetrack_dir = Path("./data/eyetracking")
    if eyetrack_dir.exists():
        for csv_file in eyetrack_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            summary_data.append({
                'dataset': csv_file.stem,
                'samples': len(df),
                'type': 'eyetracking',
                'modality': 'gaze'
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("./data/dataset_summary.csv", index=False)
    
    print("\nDataset Summary:")
    print(summary_df.to_string(index=False))
    
    total_samples = summary_df['samples'].sum()
    print(f"\nTotal samples across all datasets: {total_samples:,}")

def main():
    print("VitalLens Multi-Modal Dataset Downloader")
    print("=" * 50)
    
    os.makedirs("./data", exist_ok=True)
    
    download_fer2013()
    download_hf_emotion_datasets()
    download_mcd_rppg()
    download_eyetracking_datasets()
    
    create_dataset_summary()
    
    print("\n✅ All datasets downloaded successfully!")
    print("Run 'python test_pipeline.py' to test the training pipeline.")

if __name__ == "__main__":
    main()
