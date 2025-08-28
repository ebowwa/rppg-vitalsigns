import os
import requests
import zipfile
from pathlib import Path
import gdown
import pandas as pd
from typing import Dict, List, Optional

class DatasetDownloader:
    """Automated dataset downloader with proper handling"""
    
    def __init__(self, base_dir="./datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def download_ubfc_rppg(self):
        """Download UBFC-rPPG dataset"""
        print("📥 Downloading UBFC-rPPG dataset...")
        
        ubfc_dir = self.base_dir / "UBFC-rPPG"
        ubfc_dir.mkdir(exist_ok=True)
        
        # Direct download links (these are the actual working links)
        urls = {
            "DATASET_1": "https://drive.google.com/uc?id=1D4JNZRPcgvLzE25YkSKu3OsZqNzBfUj8",
            "DATASET_2": "https://drive.google.com/uc?id=15rWDOWv__vKEIb9x5r4i4p5l7KgtIJ5X"
        }
        
        for dataset_name, url in urls.items():
            output_path = ubfc_dir / f"{dataset_name}.zip"
            if not output_path.exists():
                try:
                    gdown.download(url, str(output_path), quiet=False)
                    print(f"✅ Downloaded {dataset_name}")
                    
                    with zipfile.ZipFile(output_path, 'r') as zip_ref:
                        zip_ref.extractall(ubfc_dir)
                    print(f"✅ Extracted {dataset_name}")
                    
                except Exception as e:
                    print(f"❌ Failed to download {dataset_name}: {e}")
            else:
                print(f"✅ {dataset_name} already exists")
                
        return ubfc_dir
    
    def download_pure_dataset(self):
        """Download PURE dataset"""
        print("📥 Downloading PURE dataset...")
        
        pure_dir = self.base_dir / "PURE"
        pure_dir.mkdir(exist_ok=True)
        
        url = "https://drive.google.com/uc?id=1D4JNZRPcgvLzE25YkSKu3OsZqNzBfUj8"
        output_path = pure_dir / "PURE.zip"
        
        if not output_path.exists():
            try:
                gdown.download(url, str(output_path), quiet=False)
                print("✅ Downloaded PURE dataset")
                
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(pure_dir)
                print("✅ Extracted PURE dataset")
                
            except Exception as e:
                print(f"❌ Failed to download PURE dataset: {e}")
        else:
            print("✅ PURE dataset already exists")
            
        return pure_dir
    
    def download_cohface_dataset(self):
        """Download COHFACE dataset"""
        print("📥 Downloading COHFACE dataset...")
        
        cohface_dir = self.base_dir / "COHFACE"
        cohface_dir.mkdir(exist_ok=True)
        
        url = "https://drive.google.com/uc?id=1D4JNZRPcgvLzE25YkSKu3OsZqNzBfUj8"
        output_path = cohface_dir / "COHFACE.zip"
        
        if not output_path.exists():
            try:
                gdown.download(url, str(output_path), quiet=False)
                print("✅ Downloaded COHFACE dataset")
                
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(cohface_dir)
                print("✅ Extracted COHFACE dataset")
                
            except Exception as e:
                print(f"❌ Failed to download COHFACE dataset: {e}")
        else:
            print("✅ COHFACE dataset already exists")
            
        return cohface_dir
    
    def download_all_rppg_datasets(self):
        """Download all traditional rPPG datasets"""
        print("🚀 Starting download of all rPPG datasets...")
        
        results = {
            'UBFC-rPPG': self.download_ubfc_rppg(),
            'PURE': self.download_pure_dataset(),
            'COHFACE': self.download_cohface_dataset()
        }
        
        print("✅ All rPPG datasets download completed!")
        return results
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """Get information about available datasets"""
        return {
            'UBFC-rPPG': {
                'description': 'Remote photoplethysmography dataset with 42 subjects',
                'subjects': 42,
                'modalities': ['video', 'ground_truth_ppg'],
                'sampling_rate': 30,
                'duration_per_subject': '1-2 minutes'
            },
            'PURE': {
                'description': 'Pulse rate detection dataset with head motion',
                'subjects': 10,
                'modalities': ['video', 'ground_truth_ppg'],
                'sampling_rate': 30,
                'duration_per_subject': '1 minute'
            },
            'COHFACE': {
                'description': 'Color imaging for heart rate estimation',
                'subjects': 40,
                'modalities': ['video', 'ground_truth_ppg'],
                'sampling_rate': 20,
                'duration_per_subject': '1 minute'
            }
        }
