# Remote Photoplethysmography (rPPG) Vital Signs Estimation

An open-source implementation for estimating vital signs (heart rate, respiratory rate) from facial videos using remote photoplethysmography.

## Overview

This project aims to recreate and extend the capabilities demonstrated in VitalLens and similar rPPG applications. It provides tools for:
- Real-time heart rate and respiratory rate estimation from video
- Multiple rPPG algorithm implementations (CHROM, POS, DeepPhys, etc.)
- Comprehensive evaluation framework
- Support for diverse public datasets

## Project Structure

```
rppg-vitalsigns/
├── src/                    # Source code
│   ├── models/            # rPPG model implementations
│   ├── preprocessing/     # Video preprocessing pipeline
│   ├── evaluation/        # Evaluation metrics and tools
│   └── utils/            # Utility functions
├── datasets/              # Dataset loaders and info
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks for experiments
├── references/            # Reference papers and materials
│   └── vitallens-paper/  # VitalLens technical report
└── README.md
```

## Key Features

- **Multi-task Learning**: Combined rPPG vital signs estimation and emotion detection
- **VitalLens Architecture**: EfficientNetV2-based model with temporal processing and attention
- **Multiple Methods**: Implementation of classical (G, CHROM, POS) and learning-based (DeepPhys, MTTS-CAN) approaches
- **Dataset Support**: Loaders for UBFC-rPPG, PURE, VIPL-HR, SCAMPS, and FER2013 emotion dataset
- **Cloud Training**: Ready-to-use scripts for Modal and RunPod GPU training
- **Evaluation**: Comprehensive metrics including MAE, SNR, Pearson correlation, and emotion accuracy
- **Real-time Processing**: Optimized for live video inference

## Performance Benchmarks

Based on VitalLens paper (on VV-Medium dataset):

| Method     | HR MAE (bpm) | Pulse SNR (dB) | Inference Time (ms) |
|------------|--------------|----------------|---------------------|
| G          | 13.74        | -3.62          | 3.4                 |
| CHROM      | 7.91         | -1.69          | 4.2                 |
| POS        | 8.51         | -1.50          | 3.6                 |
| DeepPhys   | 1.51         | 6.58           | 9.8                 |
| MTTS-CAN   | 0.99         | 7.52           | 22.1                |
| VitalLens  | 0.71         | 8.56           | 18.0                |

## Datasets

### Available Public Datasets

1. **Vital Videos** (vitalvideos.org) - 900+ subjects, diverse demographics
2. **VIPL-HR** - 2,378 VIS + 752 NIR videos, various scenarios
3. **SCAMPS** (Synthetic) - 2,800 videos, 1.68M frames
4. **UBFC-rPPG** - Standard benchmark dataset
5. **PURE** - Controlled motion scenarios
6. **MMPD** - Mobile videos with diverse conditions

See `docs/datasets.md` for detailed access instructions.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rppg-vitalsigns.git
cd rppg-vitalsigns

# Install dependencies (to be added)
pip install -r requirements.txt
```

## Quick Start

### Complete Training & Mobile Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Create rich dataset collection
python scripts/create_rich_dataset.py --download-all --priority high

# Complete pipeline: Training → Core ML → iOS
python scripts/train_and_deploy.py --platform modal --epochs 50

# Or with RunPod
python scripts/train_and_deploy.py --platform runpod --epochs 50

# Manual mobile deployment
python scripts/deploy_mobile.py --checkpoint best_model.pth --enable-pruning --enable-quantization

# One-command deployment
./mobile_deployment/deploy_vitallens.sh modal path/to/checkpoint.pth
```

### Training Only
```bash
# Download datasets
python scripts/download_datasets.py --all

# Train on Modal (A100)
python scripts/train_modal.py --epochs 50 --batch-size 16

# Train on RunPod (RTX 4090)
python scripts/train_runpod.py --epochs 50 --batch-size 8

# Test pipeline
python test_pipeline.py
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --model_path best_model.pth
```

### Example Usage

```python
# Example usage for rPPG + emotion detection
from src.models.vitallens_emotion import VitalLensEmotionModel
import torch

# Initialize model
model = VitalLensEmotionModel(sequence_length=150, num_emotions=7)

# Load trained weights
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Process video sequence (batch_size, sequence_length, channels, height, width)
video_input = torch.randn(1, 150, 3, 224, 224)
outputs = model(video_input)

print(f"Heart Rate: {outputs['heart_rate'].item():.1f} bpm")
print(f"Respiratory Rate: {outputs['resp_rate'].item():.1f} bpm")
print(f"Emotion: {torch.argmax(outputs['emotion_logits'], dim=1).item()}")
```

## 📱 Mobile Deployment

### Complete iOS Pipeline
The VitalLens multi-modal model supports complete mobile deployment with optimization:

```bash
# Complete deployment pipeline
python scripts/deploy_mobile.py \
    --checkpoint best_model.pth \
    --target-size-mb 20 \
    --target-inference-ms 18 \
    --enable-pruning \
    --enable-quantization

# Automated deployment
./mobile_deployment/deploy_vitallens.sh modal path/to/checkpoint.pth
```

### Mobile Optimization Features
- **Model Compression**: Pruning + Quantization (24.5M → <20MB)
- **Performance Optimization**: <18ms inference on iPhone
- **Multi-Modal Support**: Video + Audio + Eye-tracking
- **Neural Engine**: Optimized for iOS hardware acceleration

### iOS Integration
- **Core ML Model**: `VitalLensMultiModal.mlmodel` (iOS 15+)
- **Swift Framework**: Complete integration code provided
- **Real-time Processing**: 30 FPS video processing
- **Multi-Modal Outputs**: rPPG + Emotions + Gaze coordinates

### Performance Targets
- **Inference Time**: <18ms on iPhone (Neural Engine)
- **Model Size**: <20MB (optimized from 24.5M parameters)
- **Accuracy**: Maintains training performance after optimization
- **Battery Usage**: Optimized for mobile power constraints

## Research Context

This project builds upon extensive research in rPPG, including:
- Classical methods: G (Verkruysse et al., 2008), CHROM (de Haan & Jeanne, 2013), POS (Wang et al., 2017)
- Deep learning approaches: DeepPhys (Chen & McDuff, 2018), MTTS-CAN (Liu et al., 2020)
- Recent advances: VitalLens (Rouast, 2023)

## Contributing

Contributions are welcome! Areas of interest:
- Implementing additional rPPG algorithms
- Adding dataset loaders
- Improving real-time performance
- Enhancing robustness to movement and lighting variations

## License

MIT License (see LICENSE file)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rouast2023vitallens,
  title={VitalLens: Take A Vital Selfie},
  author={Rouast, Philipp V.},
  year={2023}
}
```

## Acknowledgments

- VitalLens paper authors for technical insights
- Public dataset contributors
- Open-source rPPG community

## Disclaimer

This is a research implementation. Not intended for medical diagnosis or clinical use.
