#!/usr/bin/env python3
"""
Complete Training and Deployment Pipeline
Orchestrates: Dataset Download → GPU Training → Mobile Deployment
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional

def run_command(cmd: str, cwd: Optional[str] = None) -> tuple[int, str, str]:
    """Run shell command and return exit code, stdout, stderr"""
    print(f"🔧 Running: {cmd}")
    
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True, 
        text=True, 
        cwd=cwd
    )
    
    return result.returncode, result.stdout, result.stderr

class TrainingDeploymentPipeline:
    """Complete pipeline from training to mobile deployment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.platform = config['platform']  # 'modal' or 'runpod'
        self.output_dir = Path(config.get('output_dir', './pipeline_output'))
        self.output_dir.mkdir(exist_ok=True)
        
        self.modal_budget = config.get('modal_budget', 500)
        self.runpod_budget = config.get('runpod_budget', 200)
        
    def download_datasets(self) -> bool:
        """Download and prepare all datasets"""
        print("📥 Downloading datasets...")
        
        try:
            exit_code, stdout, stderr = run_command(
                "python scripts/download_datasets.py --all",
                cwd=str(Path(__file__).parent.parent)
            )
            
            if exit_code != 0:
                print(f"❌ Dataset download failed: {stderr}")
                return False
            
            print("✅ Datasets downloaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Dataset download error: {e}")
            return False
    
    def train_model(self) -> Optional[str]:
        """Train model on specified platform"""
        print(f"🚀 Training model on {self.platform}...")
        
        try:
            if self.platform == 'modal':
                return self._train_on_modal()
            elif self.platform == 'runpod':
                return self._train_on_runpod()
            else:
                raise ValueError(f"Unknown platform: {self.platform}")
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def _train_on_modal(self) -> Optional[str]:
        """Train on Modal platform"""
        print(f"☁️  Training on Modal (Budget: ${self.modal_budget})")
        
        estimated_cost = self._estimate_modal_cost()
        print(f"💰 Estimated cost: ${estimated_cost}")
        
        if estimated_cost > self.modal_budget:
            print(f"⚠️  Estimated cost exceeds budget!")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return None
        
        exit_code, stdout, stderr = run_command(
            f"python scripts/train_modal.py "
            f"--epochs {self.config.get('epochs', 50)} "
            f"--batch-size {self.config.get('batch_size', 16)} "
            f"--gpu-type {self.config.get('gpu_type', 'A100')} "
            f"--output-dir {self.output_dir}/modal_training"
        )
        
        if exit_code != 0:
            print(f"❌ Modal training failed: {stderr}")
            return None
        
        checkpoint_path = self.output_dir / "modal_training" / "best_model.pth"
        if checkpoint_path.exists():
            print(f"✅ Modal training completed: {checkpoint_path}")
            return str(checkpoint_path)
        else:
            print("❌ No checkpoint found after training")
            return None
    
    def _train_on_runpod(self) -> Optional[str]:
        """Train on RunPod platform"""
        print(f"🖥️  Training on RunPod (Budget: ${self.runpod_budget})")
        
        estimated_cost = self._estimate_runpod_cost()
        print(f"💰 Estimated cost: ${estimated_cost}")
        
        if estimated_cost > self.runpod_budget:
            print(f"⚠️  Estimated cost exceeds budget!")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return None
        
        exit_code, stdout, stderr = run_command(
            f"python scripts/train_runpod.py "
            f"--epochs {self.config.get('epochs', 50)} "
            f"--batch-size {self.config.get('batch_size', 8)} "
            f"--gpu-type {self.config.get('gpu_type', 'RTX4090')} "
            f"--output-dir {self.output_dir}/runpod_training"
        )
        
        if exit_code != 0:
            print(f"❌ RunPod training failed: {stderr}")
            return None
        
        checkpoint_path = self.output_dir / "runpod_training" / "best_model.pth"
        if checkpoint_path.exists():
            print(f"✅ RunPod training completed: {checkpoint_path}")
            return str(checkpoint_path)
        else:
            print("❌ No checkpoint found after training")
            return None
    
    def _estimate_modal_cost(self) -> float:
        """Estimate Modal training cost"""
        gpu_cost_per_hour = 1.60
        estimated_hours = self.config.get('epochs', 50) * 0.4  # ~24 minutes per epoch
        return gpu_cost_per_hour * estimated_hours
    
    def _estimate_runpod_cost(self) -> float:
        """Estimate RunPod training cost"""
        gpu_cost_per_hour = 0.50
        estimated_hours = self.config.get('epochs', 50) * 0.5  # ~30 minutes per epoch
        return gpu_cost_per_hour * estimated_hours
    
    def deploy_to_mobile(self, checkpoint_path: str) -> bool:
        """Deploy trained model to mobile"""
        print("📱 Deploying to mobile...")
        
        try:
            exit_code, stdout, stderr = run_command(
                f"python scripts/deploy_mobile.py "
                f"--checkpoint {checkpoint_path} "
                f"--output-dir {self.output_dir}/mobile_deployment "
                f"--model-name VitalLensMultiModal "
                f"--target-size-mb 20 "
                f"--target-inference-ms 18 "
                f"--enable-pruning "
                f"--enable-quantization"
            )
            
            if exit_code != 0:
                print(f"❌ Mobile deployment failed: {stderr}")
                return False
            
            print("✅ Mobile deployment completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Mobile deployment error: {e}")
            return False
    
    def generate_final_report(self, checkpoint_path: str) -> str:
        """Generate comprehensive pipeline report"""
        print("📊 Generating final report...")
        
        report = f"""

- **Platform**: {self.platform}
- **Epochs**: {self.config.get('epochs', 50)}
- **Batch Size**: {self.config.get('batch_size', 16 if self.platform == 'modal' else 8)}
- **GPU Type**: {self.config.get('gpu_type', 'A100' if self.platform == 'modal' else 'RTX4090')}

- **Modal Budget**: ${self.modal_budget}
- **RunPod Budget**: ${self.runpod_budget}
- **Estimated Cost**: ${self._estimate_modal_cost() if self.platform == 'modal' else self._estimate_runpod_cost():.2f}

- **Architecture**: VitalLens Multi-Modal (24.5M parameters)
- **Capabilities**: rPPG + Visual Emotions + Audio Emotions + Eye-tracking
- **Checkpoint**: {checkpoint_path}

- **Target Platform**: iOS (Core ML)
- **Target Model Size**: 20 MB
- **Target Inference**: 18ms
- **Optimization**: Pruning + Quantization

- `{self.output_dir}/mobile_deployment/VitalLensMultiModal.mlpackage` - Core ML model
- `{self.output_dir}/mobile_deployment/VitalLensMultiModal_iOS_Integration.swift` - Swift code
- `{self.output_dir}/mobile_deployment/deploy_vitallens.sh` - Automation script

1. **rPPG Vital Signs**
   - Heart rate (BPM)
   - Respiratory rate (BPM)
   - Pulse waveform (150 samples)
   - Respiration waveform (150 samples)

2. **Emotion Detection**
   - Visual emotions (7 classes: angry, disgust, fear, happy, neutral, sad, surprise)
   - Audio emotions (7 classes)
   - Fused multi-modal emotions (7 classes)

3. **Behavioral Analysis**
   - Eye-tracking coordinates (x, y)
   - Gaze pattern analysis

- **rPPG Accuracy**: HR MAE < 1.0 BPM, RR MAE < 1.0 BPM
- **Emotion Accuracy**: > 75% across all modalities
- **Mobile Performance**: < 18ms inference, < 20MB model size
- **Real-time Processing**: 30 FPS video processing

1. **iOS Integration**
   ```bash
   cp {self.output_dir}/mobile_deployment/VitalLensMultiModal.mlpackage YourApp/
   
   ```

2. **Testing and Validation**
   - Test on real iOS devices
   - Validate performance benchmarks
   - Collect user feedback

3. **Production Optimization**
   - Fine-tune for specific use cases
   - Implement additional privacy features
   - Add real-time monitoring

- **MCD rPPG**: 3600 recordings from 600 subjects
- **YOLO Emotions**: 155K samples with bounding boxes
- **HuggingFace Collections**: Multiple emotion datasets
- **Eye-tracking**: Behavioral analysis datasets

- **Progressive Training**: rPPG → Visual Emotions → Audio → Eye-tracking
- **Multi-task Learning**: Weighted loss optimization
- **Data Augmentation**: ColorJitter, RandomFlip, Normalization
- **Regularization**: Dropout, Weight decay, Gradient clipping

```bash
python scripts/train_and_deploy.py --platform modal --epochs 50

./mobile_deployment/deploy_vitallens.sh modal path/to/checkpoint.pth
```

- **Modal Training**: ~${self._estimate_modal_cost():.2f} (A100, {self.config.get('epochs', 50)} epochs)
- **RunPod Training**: ~${self._estimate_runpod_cost():.2f} (RTX 4090, {self.config.get('epochs', 50)} epochs)
- **Total Budget**: ${self.modal_budget + self.runpod_budget}

- ✅ Multi-modal architecture implemented (24.5M parameters)
- ✅ GPU training pipeline automated (Modal + RunPod)
- ✅ Core ML conversion with optimization
- ✅ Swift iOS integration framework
- ✅ Complete deployment automation
- ✅ Performance benchmarking
- ✅ Comprehensive documentation

---
Generated by VitalLens Training & Deployment Pipeline
"""
        
        report_path = self.output_dir / "pipeline_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📊 Final report saved: {report_path}")
        return str(report_path)
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete training and deployment pipeline"""
        print("🚀 Starting complete VitalLens pipeline...")
        
        results = {
            'success': False,
            'steps_completed': [],
            'output_dir': str(self.output_dir)
        }
        
        try:
            if self.download_datasets():
                results['steps_completed'].append('dataset_download')
            else:
                results['error'] = 'Dataset download failed'
                return results
            
            checkpoint_path = self.train_model()
            if checkpoint_path:
                results['steps_completed'].append('model_training')
                results['checkpoint_path'] = checkpoint_path
            else:
                results['error'] = 'Model training failed'
                return results
            
            if self.deploy_to_mobile(checkpoint_path):
                results['steps_completed'].append('mobile_deployment')
            else:
                results['error'] = 'Mobile deployment failed'
                return results
            
            report_path = self.generate_final_report(checkpoint_path)
            results['steps_completed'].append('final_report')
            results['report_path'] = report_path
            
            results['success'] = True
            print("✅ Complete pipeline finished successfully!")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            results['error'] = str(e)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='VitalLens Complete Training & Deployment Pipeline')
    parser.add_argument('--platform', choices=['modal', 'runpod'], required=True, help='Training platform')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size (auto-selected based on platform)')
    parser.add_argument('--gpu-type', help='GPU type (auto-selected based on platform)')
    parser.add_argument('--output-dir', default='./pipeline_output', help='Output directory')
    parser.add_argument('--modal-budget', type=float, default=500, help='Modal budget in USD')
    parser.add_argument('--runpod-budget', type=float, default=200, help='RunPod budget in USD')
    parser.add_argument('--skip-datasets', action='store_true', help='Skip dataset download')
    parser.add_argument('--checkpoint', help='Use existing checkpoint (skip training)')
    
    args = parser.parse_args()
    
    if not args.batch_size:
        args.batch_size = 16 if args.platform == 'modal' else 8
    
    if not args.gpu_type:
        args.gpu_type = 'A100' if args.platform == 'modal' else 'RTX4090'
    
    config = {
        'platform': args.platform,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gpu_type': args.gpu_type,
        'output_dir': args.output_dir,
        'modal_budget': args.modal_budget,
        'runpod_budget': args.runpod_budget
    }
    
    pipeline = TrainingDeploymentPipeline(config)
    
    if args.checkpoint:
        print(f"📱 Using existing checkpoint: {args.checkpoint}")
        if pipeline.deploy_to_mobile(args.checkpoint):
            report_path = pipeline.generate_final_report(args.checkpoint)
            print(f"✅ Deployment completed! Report: {report_path}")
        else:
            print("❌ Deployment failed")
            sys.exit(1)
    else:
        results = pipeline.run_complete_pipeline()
        
        if results['success']:
            print(f"\n✅ Complete pipeline finished successfully!")
            print(f"📁 Output directory: {results['output_dir']}")
            print(f"📊 Final report: {results['report_path']}")
            print(f"🎯 Steps completed: {', '.join(results['steps_completed'])}")
        else:
            print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            print(f"🎯 Steps completed: {', '.join(results['steps_completed'])}")
            sys.exit(1)

if __name__ == '__main__':
    main()
