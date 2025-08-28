import os
import zipfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import requests
import json
import logging

class KaggleDatasetDownloader:
    """Kaggle dataset downloader with API integration"""
    
    def __init__(self, base_dir="./data/kaggle_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.kaggle_available = self._check_kaggle_credentials()
        
    def _check_kaggle_credentials(self) -> bool:
        """Check if Kaggle API credentials are available"""
        try:
            import kaggle
            return True
        except ImportError:
            logging.warning("Kaggle API not installed. Install with: pip install kaggle")
            return False
        except OSError as e:
            if "credentials" in str(e).lower():
                logging.warning("Kaggle credentials not found. Please set up ~/.kaggle/kaggle.json")
                return False
            return False
    
    def download_dataset(self, dataset_id: str, download_path: Optional[str] = None) -> Path:
        """
        Download a Kaggle dataset
        
        Args:
            dataset_id: Kaggle dataset identifier (e.g., "username/dataset-name")
            download_path: Optional custom download path
            
        Returns:
            Path to downloaded dataset directory
        """
        if not self.kaggle_available:
            raise RuntimeError("Kaggle API not available. Please install and configure kaggle package.")
        
        import kaggle
        
        if download_path is None:
            download_path = self.base_dir / dataset_id.replace('/', '_')
        else:
            download_path = Path(download_path)
        
        download_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Downloading Kaggle dataset: {dataset_id}")
        print(f"   Destination: {download_path}")
        
        try:
            kaggle.api.dataset_download_files(
                dataset_id,
                path=str(download_path),
                unzip=True
            )
            
            print(f"✅ Successfully downloaded: {dataset_id}")
            return download_path
            
        except Exception as e:
            print(f"❌ Failed to download {dataset_id}: {e}")
            raise
    
    def download_audio_emotion_dataset(self) -> Path:
        """Download audio emotion dataset from Kaggle"""
        dataset_id = "ejlok1/audio-emotion-part-1-explore-data"
        return self.download_dataset(dataset_id)
    
    def download_facial_emotion_dataset(self) -> Path:
        """Download facial emotion dataset from Kaggle"""
        dataset_id = "mh0386/facial-emotions-detection"
        return self.download_dataset(dataset_id)
    
    def download_fer2013_dataset(self) -> Path:
        """Download FER2013 emotion dataset"""
        dataset_id = "msambare/fer2013"
        return self.download_dataset(dataset_id)
    
    def list_available_datasets(self) -> List[Dict]:
        """List available emotion-related datasets on Kaggle"""
        if not self.kaggle_available:
            return []
        
        import kaggle
        
        try:
            datasets = kaggle.api.dataset_list(search="emotion", page_size=20)
            
            dataset_info = []
            for dataset in datasets:
                dataset_info.append({
                    'id': f"{dataset.ownerName}/{dataset.datasetName}",
                    'title': dataset.title,
                    'size': dataset.totalBytes,
                    'download_count': dataset.downloadCount,
                    'vote_count': dataset.voteCount,
                    'created': dataset.creationDate
                })
            
            return dataset_info
            
        except Exception as e:
            print(f"Error listing datasets: {e}")
            return []
    
    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get information about a specific dataset"""
        if not self.kaggle_available:
            return {}
        
        import kaggle
        
        try:
            dataset = kaggle.api.dataset_view(dataset_id)
            
            return {
                'id': dataset_id,
                'title': dataset.title,
                'description': dataset.description,
                'size': dataset.totalBytes,
                'file_count': len(dataset.files),
                'download_count': dataset.downloadCount,
                'vote_count': dataset.voteCount,
                'tags': dataset.tags,
                'license': dataset.licenseName,
                'created': dataset.creationDate,
                'updated': dataset.lastUpdated
            }
            
        except Exception as e:
            print(f"Error getting dataset info: {e}")
            return {}
    
    def download_multiple_datasets(self, dataset_ids: List[str]) -> Dict[str, Path]:
        """Download multiple datasets"""
        results = {}
        
        for dataset_id in dataset_ids:
            try:
                path = self.download_dataset(dataset_id)
                results[dataset_id] = path
            except Exception as e:
                print(f"Failed to download {dataset_id}: {e}")
                results[dataset_id] = None
        
        return results
    
    def create_dataset_summary(self, dataset_paths: Dict[str, Path]) -> pd.DataFrame:
        """Create a summary of downloaded datasets"""
        summary_data = []
        
        for dataset_id, path in dataset_paths.items():
            if path is None or not path.exists():
                continue
            
            file_count = len(list(path.glob('**/*')))
            
            total_size = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
            
            summary_data.append({
                'dataset_id': dataset_id,
                'path': str(path),
                'file_count': file_count,
                'size_mb': total_size / (1024 * 1024),
                'status': 'downloaded'
            })
        
        return pd.DataFrame(summary_data)
