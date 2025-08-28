# 📱 Mobile Deployment Guide

Complete end-to-end pipeline for deploying VitalLens multi-modal models to iPhone edge devices.

## Overview

The VitalLens mobile deployment pipeline automates the complete workflow from Modal/RunPod GPU training to iPhone edge deployment, including model optimization for mobile constraints.

### Architecture Support
- **Multi-Modal Inputs**: Video (150 frames) + Audio (spectrograms) + Eye-tracking coordinates
- **Multi-Modal Outputs**: rPPG waveforms + Heart/Respiratory rates + Emotion logits + Gaze coordinates
- **Model Size**: 24.5M parameters → <20MB optimized for mobile
- **Performance**: <18ms inference on iPhone Neural Engine

## Quick Start

### Complete Pipeline
```bash
# Complete training → Core ML → iOS deployment
python scripts/train_and_deploy.py --platform modal --epochs 50

# Or with RunPod for cost optimization
python scripts/train_and_deploy.py --platform runpod --epochs 50

# Manual mobile deployment from checkpoint
python scripts/deploy_mobile.py \
    --checkpoint best_model.pth \
    --enable-pruning \
    --enable-quantization \
    --target-size-mb 20 \
    --target-inference-ms 18
```

### One-Command Deployment
```bash
# Automated deployment script
./mobile_deployment/deploy_vitallens.sh modal path/to/checkpoint.pth
```

## Mobile Optimization

### Model Compression Techniques
1. **Structured Pruning**: Remove 30-50% of less critical parameters
2. **Dynamic Quantization**: INT8 inference for 4x memory reduction
3. **Knowledge Distillation**: Optional student model with EfficientNet-B0

### Core ML Conversion
- **Format**: `.mlpackage` (ML Program) for iOS 15+
- **Compute Units**: Neural Engine + GPU + CPU optimization
- **Input Types**: Multi-modal tensor specifications
- **Output Types**: 8 output tensors for complete multi-modal results

### Performance Targets
- **Inference Time**: <18ms on iPhone (Neural Engine)
- **Model Size**: <20MB (optimized from 24.5M parameters)
- **Accuracy**: Maintains training performance after optimization
- **Battery Usage**: Optimized for mobile power constraints

## iOS Integration

### Core ML Model Usage
```swift
import CoreML
import Vision

// Load the optimized VitalLens model
guard let model = try? VitalLensMultiModal(configuration: MLModelConfiguration()) else {
    fatalError("Failed to load Core ML model")
}

// Prepare multi-modal inputs
let videoInput = MLMultiArray(/* 150 video frames */)
let audioInput = MLMultiArray(/* audio spectrogram */)
let eyetrackInput = MLMultiArray(/* gaze coordinates */)

// Run inference
let prediction = try model.prediction(
    video_frames: videoInput,
    audio_features: audioInput,
    eyetrack_coords: eyetrackInput
)

// Extract results
let heartRate = prediction.heart_rate[0].floatValue
let respiratoryRate = prediction.resp_rate[0].floatValue
let emotionLogits = prediction.emotion_logits
let gazeCoords = prediction.eyetrack_coordinates
```

### Swift Integration Features
- **Real-time Processing**: 30 FPS video processing
- **Multi-Modal Support**: Handles all input modalities
- **Emotion Classification**: 7 emotion classes with confidence scores
- **Performance Monitoring**: Built-in inference time tracking

## Platform-Specific Training

### Modal (A100 GPU)
```bash
# High-performance training with gradient accumulation
python scripts/train_modal.py \
    --epochs 50 \
    --batch-size 16 \
    --gpu a100 \
    --auto-deploy-mobile
```
- **Cost**: ~$16-32 for full training
- **Performance**: Faster training, larger batch sizes
- **Memory**: 40GB A100 for complex multi-modal models

### RunPod (RTX 4090)
```bash
# Cost-effective training with mixed precision
python scripts/train_runpod.py \
    --epochs 50 \
    --batch-size 8 \
    --mixed-precision \
    --auto-deploy-mobile
```
- **Cost**: ~$10 for full training
- **Performance**: Cost-optimized, good for experimentation
- **Memory**: 24GB RTX 4090 with efficient memory usage

## Deployment Automation

### Automated Pipeline Features
- **Budget Tracking**: Cost estimation for Modal ($500) and RunPod ($200)
- **Model Validation**: Automatic size and performance checking
- **iOS Project Generation**: Ready-to-use Xcode project template
- **Performance Benchmarking**: Inference time and accuracy validation

### Generated Artifacts
```
mobile_deployment/
├── VitalLensMultiModal.mlpackage          # Core ML model
├── VitalLensMultiModal_iOS_Integration.swift  # Swift integration
├── deploy_vitallens.sh                    # Automation script
├── iOS_Project/                           # Xcode project template
└── deployment_report.md                   # Performance report
```

## Performance Benchmarking

### Validation Metrics
- **Model Size**: Verify <20MB target
- **Inference Time**: Measure on iOS device
- **Accuracy Preservation**: Compare with original model
- **Memory Usage**: Monitor during inference

### Benchmark Results
```bash
# Run performance benchmark
python scripts/deploy_mobile.py --benchmark-only --model-path VitalLensMultiModal.mlpackage

# Expected results:
# ✅ Model size: 18.5 MB (target: 20 MB)
# ✅ Inference time: 16ms (target: 18ms)
# ✅ Accuracy preserved: 98.5% of original
# ✅ Memory usage: 45MB peak
```

## Troubleshooting

### Common Issues
1. **Core ML Conversion Errors**
   - Ensure PyTorch model uses supported operations
   - Check tensor shapes and data types
   - Use `.mlpackage` format for ML Program models

2. **Model Size Exceeds Target**
   - Increase pruning percentage
   - Apply more aggressive quantization
   - Consider knowledge distillation

3. **Inference Time Too Slow**
   - Optimize for Neural Engine compute units
   - Reduce model complexity
   - Check input preprocessing efficiency

### Debug Commands
```bash
# Test Core ML conversion only
python scripts/deploy_mobile.py --test-conversion-only

# Validate Swift integration syntax
python scripts/deploy_mobile.py --test-swift-only

# Check model optimization results
python scripts/deploy_mobile.py --optimization-report
```

## Advanced Features

### Custom Optimization
```python
# Custom pruning configuration
from scripts.deploy_mobile import MobileDeploymentPipeline

pipeline = MobileDeploymentPipeline(
    model_name="VitalLensCustom",
    target_size_mb=15,  # More aggressive target
    target_inference_ms=12,  # Faster inference
    pruning_ratio=0.6,  # 60% parameter reduction
    quantization_bits=8  # INT8 quantization
)
```

### Multi-Platform Support
- **iOS**: Primary target with Neural Engine optimization
- **Android**: TensorFlow Lite conversion (future)
- **Web**: ONNX.js deployment (future)

## Cost Analysis

### Training Costs
- **Modal A100**: $16-32 for 50 epochs (high performance)
- **RunPod RTX 4090**: $10 for 50 epochs (cost-effective)
- **Total Budget**: $700 available ($500 Modal + $200 RunPod)

### Deployment Efficiency
- **One-time Setup**: ~2 hours for complete pipeline
- **Iteration Time**: ~30 minutes for model updates
- **Testing Cycle**: ~5 minutes for validation

## Next Steps

1. **Deploy to iOS Device**: Test on actual iPhone hardware
2. **Performance Optimization**: Fine-tune for specific use cases
3. **User Experience**: Integrate with app UI/UX
4. **Monitoring**: Add analytics and performance tracking

For detailed implementation examples, see the generated iOS project template and Swift integration code.
