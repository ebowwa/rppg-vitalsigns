#!/usr/bin/env python3
"""
Complete Data Pipeline Automation Script

Automates the entire data handling workflow:
1. Download all datasets from registry
2. Validate data integrity
3. Preprocess and cache data
4. Generate training-ready datasets
5. Create performance benchmarks

Usage:
    python scripts/data_pipeline_automation.py --download-all --preprocess --validate
    python scripts/data_pipeline_automation.py --dataset "boltuix/emotions-dataset" --quick-setup
    python scripts/data_pipeline_automation.py --benchmark --report
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
import cv2
import librosa
import pickle
from tqdm import tqdm

from src.data.dataset import RPPGEmotionDataset
from scripts.create_rich_dataset import RichDatasetCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPipelineAutomation:
    """Complete automation for VitalLens data pipeline"""
    
    def __init__(self, data_dir: str = "./data", cache_dir: str = "./data/cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.registry_path = Path("rich_datasets/dataset_registry.json")
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry = self._load_registry()
        
        self.rich_collector = RichDatasetCollector()
        
        logger.info(f"Initialized DataPipelineAutomation with {len(self.registry['papers'])} datasets")
    
    def _load_registry(self) -> Dict:
        """Load dataset registry"""
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Registry file not found: {self.registry_path}")
            return {"neurips_research_registry": {"papers": []}}
    
    def download_all_datasets(self, priority_only: bool = False) -> Dict[str, bool]:
        """Download all datasets from registry"""
        logger.info("Starting dataset download process...")
        
        download_results = {}
        datasets_to_download = []
        
        if "neurips_research_registry" in self.registry:
            dataset_registry_path = Path("rich_datasets/dataset_registry.json")
            if dataset_registry_path.exists():
                with open(dataset_registry_path, 'r') as f:
                    dataset_registry = json.load(f)
                    datasets_to_download = dataset_registry.get("datasets", [])
        
        if not datasets_to_download:
            datasets_to_download = [
                {"name": "boltuix/emotions-dataset", "source": "huggingface", "priority": "high"},
                {"name": "ChristophSchuhmann/emotions", "source": "huggingface", "priority": "high"},
                {"name": "AdamCodd/yolo-emotions", "source": "huggingface", "priority": "high"},
                {"name": "kyegorov/mcd_rppg", "source": "huggingface", "priority": "high"},
                {"name": "julienmercier/eyetracking", "source": "huggingface", "priority": "medium"}
            ]
        
        for dataset_info in datasets_to_download:
            if priority_only and dataset_info.get("priority") != "high":
                continue
                
            dataset_name = dataset_info["name"]
            source = dataset_info.get("source", "huggingface")
            
            logger.info(f"Downloading {dataset_name} from {source}...")
            
            try:
                if source == "huggingface":
                    success = self._download_huggingface_dataset(dataset_name)
                elif source == "kaggle":
                    success = self._download_kaggle_dataset(dataset_name)
                else:
                    logger.warning(f"Unknown source {source} for {dataset_name}")
                    success = False
                
                download_results[dataset_name] = success
                
            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {e}")
                download_results[dataset_name] = False
        
        successful = sum(download_results.values())
        total = len(download_results)
        logger.info(f"Download complete: {successful}/{total} datasets successful")
        
        return download_results
    
    def _download_huggingface_dataset(self, dataset_name: str) -> bool:
        """Download HuggingFace dataset"""
        try:
            cache_path = self.data_dir / "huggingface_cache" / dataset_name.replace("/", "_")
            cache_path.mkdir(parents=True, exist_ok=True)
            
            dataset = load_dataset(
                dataset_name,
                cache_dir=str(cache_path),
                trust_remote_code=True
            )
            
            logger.info(f"Successfully downloaded {dataset_name}")
            logger.info(f"Dataset info: {dataset}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return False
    
    def _download_kaggle_dataset(self, dataset_name: str) -> bool:
        """Download Kaggle dataset"""
        try:
            logger.warning(f"Kaggle download not implemented for {dataset_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset {dataset_name}: {e}")
            return False
    
    def validate_datasets(self) -> Dict[str, Dict]:
        """Validate all downloaded datasets"""
        logger.info("Starting dataset validation...")
        
        validation_results = {}
        
        hf_cache_dir = self.data_dir / "huggingface_cache"
        if hf_cache_dir.exists():
            for dataset_dir in hf_cache_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name.replace("_", "/")
                    logger.info(f"Validating {dataset_name}...")
                    
                    validation_results[dataset_name] = self._validate_single_dataset(dataset_name)
        
        valid_datasets = sum(1 for result in validation_results.values() if result.get("valid", False))
        total_datasets = len(validation_results)
        logger.info(f"Validation complete: {valid_datasets}/{total_datasets} datasets valid")
        
        return validation_results
    
    def _validate_single_dataset(self, dataset_name: str) -> Dict:
        """Validate a single dataset"""
        validation_result = {
            "dataset_name": dataset_name,
            "valid": False,
            "total_samples": 0,
            "valid_samples": 0,
            "errors": []
        }
        
        try:
            dataset = load_dataset(dataset_name, trust_remote_code=True)
            
            if "train" in dataset:
                train_data = dataset["train"]
                validation_result["total_samples"] = len(train_data)
                
                sample_size = min(100, len(train_data))
                valid_samples = 0
                
                for i in range(sample_size):
                    try:
                        sample = train_data[i]
                        
                        if self._validate_sample(sample):
                            valid_samples += 1
                    except Exception as e:
                        validation_result["errors"].append(f"Sample {i}: {str(e)}")
                
                validation_result["valid_samples"] = valid_samples
                validation_result["sample_validity_rate"] = valid_samples / sample_size
                validation_result["valid"] = validation_result["sample_validity_rate"] > 0.8
            
        except Exception as e:
            validation_result["errors"].append(f"Dataset loading error: {str(e)}")
        
        return validation_result
    
    def _validate_sample(self, sample: Dict) -> bool:
        """Validate a single sample"""
        try:
            if "image" in sample:
                image = sample["image"]
                if image is None:
                    return False
                
                if "emotion" in sample:
                    emotion = sample["emotion"]
                    if not isinstance(emotion, int) or not (0 <= emotion <= 6):
                        return False
            
            elif "video" in sample:
                video_path = sample["video"]
                if not isinstance(video_path, str) or not video_path:
                    return False
            
            elif "audio" in sample:
                audio_data = sample["audio"]
                if audio_data is None:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def preprocess_datasets(self, num_workers: int = 4) -> Dict[str, bool]:
        """Preprocess all datasets for training"""
        logger.info("Starting dataset preprocessing...")
        
        preprocessing_results = {}
        
        hf_cache_dir = self.data_dir / "huggingface_cache"
        if hf_cache_dir.exists():
            for dataset_dir in hf_cache_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name.replace("_", "/")
                    logger.info(f"Preprocessing {dataset_name}...")
                    
                    try:
                        success = self._preprocess_single_dataset(dataset_name, num_workers)
                        preprocessing_results[dataset_name] = success
                    except Exception as e:
                        logger.error(f"Failed to preprocess {dataset_name}: {e}")
                        preprocessing_results[dataset_name] = False
        
        successful = sum(preprocessing_results.values())
        total = len(preprocessing_results)
        logger.info(f"Preprocessing complete: {successful}/{total} datasets successful")
        
        return preprocessing_results
    
    def _preprocess_single_dataset(self, dataset_name: str, num_workers: int) -> bool:
        """Preprocess a single dataset"""
        try:
            dataset = load_dataset(dataset_name, trust_remote_code=True)
            
            if "train" not in dataset:
                logger.warning(f"No train split found for {dataset_name}")
                return False
            
            train_data = dataset["train"]
            
            cache_dir = self.cache_dir / dataset_name.replace("/", "_")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            if num_workers > 1:
                with mp.Pool(num_workers) as pool:
                    results = pool.starmap(
                        self._preprocess_sample,
                        [(sample, i, cache_dir) for i, sample in enumerate(train_data)]
                    )
            else:
                results = []
                for i, sample in enumerate(tqdm(train_data, desc=f"Preprocessing {dataset_name}")):
                    result = self._preprocess_sample(sample, i, cache_dir)
                    results.append(result)
            
            successful_samples = sum(results)
            total_samples = len(results)
            
            logger.info(f"Preprocessed {successful_samples}/{total_samples} samples for {dataset_name}")
            
            return successful_samples > 0
            
        except Exception as e:
            logger.error(f"Error preprocessing {dataset_name}: {e}")
            return False
    
    def _preprocess_sample(self, sample: Dict, sample_id: int, cache_dir: Path) -> bool:
        """Preprocess a single sample"""
        try:
            cache_file = cache_dir / f"sample_{sample_id}.pkl"
            
            if cache_file.exists():
                return True
            
            processed_sample = {}
            
            if "image" in sample:
                image = sample["image"]
                if image is not None:
                    import torchvision.transforms as transforms
                    
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])
                    
                    processed_sample["image"] = transform(image)
                    
                    if "emotion" in sample:
                        processed_sample["emotion"] = sample["emotion"]
            
            elif "video" in sample:
                processed_sample["video_path"] = sample["video"]
                if "ppg" in sample:
                    processed_sample["ppg"] = sample["ppg"]
            
            elif "audio" in sample:
                processed_sample["audio_path"] = sample["audio"]
            
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_sample, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing sample {sample_id}: {e}")
            return False
    
    def create_training_datasets(self) -> Dict[str, str]:
        """Create training-ready datasets"""
        logger.info("Creating training-ready datasets...")
        
        dataset_paths = {}
        
        try:
            dataset = RPPGEmotionDataset(
                data_dir=str(self.data_dir),
                sequence_length=150,
                include_audio=True,
                include_eyetracking=True,
                emotion_classes=7
            )
            
            config_path = self.data_dir / "training_dataset_config.json"
            config = {
                "data_dir": str(self.data_dir),
                "sequence_length": 150,
                "include_audio": True,
                "include_eyetracking": True,
                "emotion_classes": 7,
                "total_samples": len(dataset) if hasattr(dataset, '__len__') else "unknown"
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            dataset_paths["multimodal_dataset"] = str(config_path)
            logger.info(f"Created training dataset configuration: {config_path}")
            
        except Exception as e:
            logger.error(f"Error creating training datasets: {e}")
        
        return dataset_paths
    
    def benchmark_performance(self) -> Dict[str, float]:
        """Benchmark data loading performance"""
        logger.info("Benchmarking data loading performance...")
        
        benchmarks = {}
        
        try:
            dataset = RPPGEmotionDataset(
                data_dir=str(self.data_dir),
                sequence_length=150,
                include_audio=True,
                include_eyetracking=True,
                emotion_classes=7
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=8,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            start_time = time.time()
            num_batches = 10
            
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                _ = batch.get('video', torch.empty(0)).shape
            
            elapsed_time = time.time() - start_time
            samples_per_second = (num_batches * dataloader.batch_size) / elapsed_time
            
            benchmarks["samples_per_second"] = samples_per_second
            benchmarks["batch_loading_time"] = elapsed_time / num_batches
            
            logger.info(f"Benchmark results: {samples_per_second:.1f} samples/sec")
            
        except Exception as e:
            logger.error(f"Error during benchmarking: {e}")
            benchmarks["error"] = str(e)
        
        return benchmarks
    
    def generate_report(self, download_results: Dict, validation_results: Dict, 
                       preprocessing_results: Dict, benchmarks: Dict) -> str:
        """Generate comprehensive pipeline report"""
        
        report_path = self.data_dir / "pipeline_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Data Pipeline Automation Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Dataset Download Summary\n\n")
            successful_downloads = sum(download_results.values())
            total_downloads = len(download_results)
            f.write(f"- **Success Rate**: {successful_downloads}/{total_downloads} ({successful_downloads/total_downloads*100:.1f}%)\n\n")
            
            for dataset, success in download_results.items():
                status = "✅" if success else "❌"
                f.write(f"- {status} {dataset}\n")
            f.write("\n")
            
            f.write("## Dataset Validation Summary\n\n")
            valid_datasets = sum(1 for result in validation_results.values() if result.get("valid", False))
            total_validated = len(validation_results)
            if total_validated > 0:
                f.write(f"- **Validation Rate**: {valid_datasets}/{total_validated} ({valid_datasets/total_validated*100:.1f}%)\n\n")
                
                for dataset, result in validation_results.items():
                    status = "✅" if result.get("valid", False) else "❌"
                    samples = result.get("total_samples", 0)
                    f.write(f"- {status} {dataset} ({samples} samples)\n")
            f.write("\n")
            
            f.write("## Preprocessing Summary\n\n")
            successful_preprocessing = sum(preprocessing_results.values())
            total_preprocessing = len(preprocessing_results)
            if total_preprocessing > 0:
                f.write(f"- **Success Rate**: {successful_preprocessing}/{total_preprocessing} ({successful_preprocessing/total_preprocessing*100:.1f}%)\n\n")
                
                for dataset, success in preprocessing_results.items():
                    status = "✅" if success else "❌"
                    f.write(f"- {status} {dataset}\n")
            f.write("\n")
            
            f.write("## Performance Benchmarks\n\n")
            if benchmarks:
                if "samples_per_second" in benchmarks:
                    f.write(f"- **Loading Speed**: {benchmarks['samples_per_second']:.1f} samples/second\n")
                if "batch_loading_time" in benchmarks:
                    f.write(f"- **Batch Loading Time**: {benchmarks['batch_loading_time']:.3f} seconds\n")
                if "error" in benchmarks:
                    f.write(f"- **Error**: {benchmarks['error']}\n")
            f.write("\n")
            
            f.write("## Recommendations\n\n")
            if successful_downloads < total_downloads:
                f.write("- Some datasets failed to download. Check network connectivity and API credentials.\n")
            if valid_datasets < total_validated:
                f.write("- Some datasets failed validation. Review data integrity and format compatibility.\n")
            if "samples_per_second" in benchmarks and benchmarks["samples_per_second"] < 10:
                f.write("- Data loading speed is slow. Consider increasing num_workers or using SSD storage.\n")
            f.write("\n")
        
        logger.info(f"Generated pipeline report: {report_path}")
        return str(report_path)

def main():
    parser = argparse.ArgumentParser(description="VitalLens Data Pipeline Automation")
    
    parser.add_argument("--download-all", action="store_true", help="Download all datasets")
    parser.add_argument("--download-priority", action="store_true", help="Download only high-priority datasets")
    parser.add_argument("--validate", action="store_true", help="Validate downloaded datasets")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess datasets")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark performance")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive report")
    
    parser.add_argument("--dataset", type=str, help="Process specific dataset")
    parser.add_argument("--quick-setup", action="store_true", help="Quick setup for single dataset")
    
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--cache-dir", type=str, default="./data/cache", help="Cache directory")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
    
    args = parser.parse_args()
    
    pipeline = DataPipelineAutomation(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir
    )
    
    download_results = {}
    validation_results = {}
    preprocessing_results = {}
    benchmarks = {}
    
    if args.download_all:
        download_results = pipeline.download_all_datasets(priority_only=False)
    elif args.download_priority:
        download_results = pipeline.download_all_datasets(priority_only=True)
    
    if args.validate:
        validation_results = pipeline.validate_datasets()
    
    if args.preprocess:
        preprocessing_results = pipeline.preprocess_datasets(num_workers=args.num_workers)
    
    if args.benchmark:
        benchmarks = pipeline.benchmark_performance()
    
    if args.quick_setup and args.dataset:
        logger.info(f"Quick setup for {args.dataset}")
        download_results = {args.dataset: pipeline._download_huggingface_dataset(args.dataset)}
        validation_results = {args.dataset: pipeline._validate_single_dataset(args.dataset)}
        preprocessing_results = {args.dataset: pipeline._preprocess_single_dataset(args.dataset, args.num_workers)}
    
    if args.report or any([download_results, validation_results, preprocessing_results, benchmarks]):
        report_path = pipeline.generate_report(
            download_results, validation_results, preprocessing_results, benchmarks
        )
        print(f"\n📊 Pipeline report generated: {report_path}")
    
    if args.preprocess or args.quick_setup:
        dataset_paths = pipeline.create_training_datasets()
        if dataset_paths:
            print(f"\n🎯 Training datasets ready: {dataset_paths}")

if __name__ == "__main__":
    main()
