# VitalLens Training Implementations

This directory contains various implementations of the VitalLens rPPG training methodology, organized by approach and complexity level.

## Directory Structure

### 📁 `complete/`
**Full-featured implementations with comprehensive pipelines**
- `VitalLens_Complete.ipynb` - Complete implementation with automated dataset downloads, proper preprocessing pipelines, VitalLens-style CNN architecture, training with multiple datasets, cross-dataset evaluation, and mobile deployment (Core ML)
- **Target Performance**: < 2.0 BPM MAE (VitalLens: 0.71 BPM)
- **Datasets**: UBFC-rPPG, PURE, COHFACE, VIPL-HR
- **Features**: ✅ Automated downloads, ✅ Preprocessing, ✅ CNN architecture, ✅ Multi-dataset training, ✅ Evaluation, ✅ Mobile deployment

### 📁 `corrected/`
**Paper-accurate implementations following VitalLens specifications exactly**
- `VitalLens_Corrected.ipynb` - Corrected implementation based on thorough analysis of the VitalLens paper
- **Key Corrections**: Proper waveform estimation, FFT-based rate extraction, variable chunk processing, focus on public datasets, quality-aware training
- **Architecture**: EfficientNetV2 backbone → waveform estimation
- **Performance**: 0.71 BPM MAE, 0.76 RR MAE on VV-Medium
- **Inference**: 18ms per frame (excluding face detection)

### 📁 `basic/`
**Simpler implementations for learning and basic use cases**
- `VitalLens_Training.ipynb` - Basic CNN-based rPPG heart rate estimation using EfficientNetV2
- **Approach**: Straightforward implementation for understanding core concepts
- **Target**: Replicating VitalLens approach with simplified pipeline
- **Good for**: Learning, prototyping, basic experiments

### 📁 `reference/`
**Reference implementations with evaluation frameworks**
- `vitallens_training.ipynb` - Comprehensive reference implementation with synthetic data and evaluation using paper's reference data
- **Features**: EfficientNetV2-based model, multi-task learning, comprehensive metrics tracking, factor analysis
- **Data Integration**: Uses actual evaluation data from research paper
- **Evaluation**: Matches paper benchmarks, includes demographic and environmental impact analysis

## Usage Guidelines

1. **For Production Use**: Start with `complete/VitalLens_Complete.ipynb`
2. **For Research/Paper Replication**: Use `corrected/VitalLens_Corrected.ipynb`
3. **For Learning**: Begin with `basic/VitalLens_Training.ipynb`
4. **For Evaluation/Benchmarking**: Use `reference/vitallens_training.ipynb`

## Dependencies

All notebooks require:
- PyTorch
- torchvision
- EfficientNetV2 models
- OpenCV
- NumPy, Pandas, Matplotlib

Specific requirements may vary by implementation. Check individual notebooks for detailed dependency lists.

## Data Requirements

- **Complete**: Downloads datasets automatically
- **Corrected**: Uses public datasets (UBFC-rPPG, PURE, COHFACE)
- **Basic**: Requires manual dataset setup
- **Reference**: Uses synthetic data + paper reference data from `../references/data/`

## Performance Targets

| Implementation | HR MAE (BPM) | RR MAE (BPM) | Notes |
|---------------|--------------|--------------|-------|
| Complete | < 2.0 | - | Multi-dataset training |
| Corrected | 0.71 | 0.76 | Paper-accurate |
| Basic | Variable | - | Learning-focused |
| Reference | 0.71 | 0.76 | Benchmark matching |
