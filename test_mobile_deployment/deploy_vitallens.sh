#!/bin/bash

set -e

PLATFORM=$1
CHECKPOINT_PATH=$2
OUTPUT_DIR="./mobile_deployment"

if [ -z "$PLATFORM" ] || [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: $0 [modal|runpod] [model_checkpoint_path]"
    exit 1
fi

echo "🚀 Starting VitalLens Multi-Modal Deployment Pipeline"
echo "Platform: $PLATFORM"
echo "Checkpoint: $CHECKPOINT_PATH"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "📱 Converting model to Core ML..."
python scripts/deploy_mobile.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "VitalLensMultiModal" \
    --target-size-mb 20 \
    --target-inference-ms 18 \
    --enable-pruning \
    --enable-quantization

echo "✅ Validating Core ML model..."
if [ -f "$OUTPUT_DIR/VitalLensMultiModal.mlpackage" ]; then
    echo "✅ Core ML model created successfully"
    
    MODEL_SIZE=$(du -m "$OUTPUT_DIR/VitalLensMultiModal.mlpackage" | cut -f1)
    echo "📊 Model size: ${MODEL_SIZE} MB"
    
    if [ "$MODEL_SIZE" -gt 20 ]; then
        echo "⚠️  Model size exceeds 20MB target. Consider additional optimization."
    fi
else
    echo "❌ Core ML model creation failed"
    exit 1
fi

echo "📱 Creating iOS project template..."
mkdir -p "$OUTPUT_DIR/iOS_Project"
cp "$OUTPUT_DIR/VitalLensMultiModal.mlpackage" "$OUTPUT_DIR/iOS_Project/"
cp "$OUTPUT_DIR/VitalLensMultiModal_iOS_Integration.swift" "$OUTPUT_DIR/iOS_Project/"

echo "📊 Generating deployment report..."
cat > "$OUTPUT_DIR/deployment_report.md" << EOF

- **Platform**: $PLATFORM
- **Source Checkpoint**: $CHECKPOINT_PATH
- **Model Size**: ${MODEL_SIZE} MB
- **Target Size**: 20 MB
- **Target Inference**: 18ms

- \`VitalLensMultiModal.mlpackage\` - Core ML model for iOS
- \`VitalLensMultiModal_iOS_Integration.swift\` - Swift integration code
- \`iOS_Project/\` - Ready-to-use iOS project template

- ✅ rPPG vital signs (heart rate, respiratory rate)
- ✅ Visual emotion detection (7 classes)
- ✅ Audio emotion recognition
- ✅ Eye-tracking integration
- ✅ Multi-modal fusion

1. Copy \`.mlpackage\` file to your Xcode project
2. Implement the Swift integration code
3. Test on iOS device (iOS 15+)
4. Optimize for your specific use case

- **Inference Time**: < 18ms on iPhone
- **Model Size**: < 20MB
- **Accuracy**: Maintain training performance
- **Battery Usage**: Optimized for mobile

\`\`\`bash
python scripts/train_modal.py --epochs 50 --batch-size 16

python scripts/train_runpod.py --epochs 50 --batch-size 8

./deploy_vitallens.sh modal path/to/best_model.pth
\`\`\`
EOF

echo "✅ Deployment pipeline completed successfully!"
echo "📁 Output directory: $OUTPUT_DIR"
echo "📊 Deployment report: $OUTPUT_DIR/deployment_report.md"
echo ""
echo "🔧 Next steps:"
echo "1. Copy VitalLensMultiModal.mlpackage to your Xcode project"
echo "2. Implement the Swift integration code"
echo "3. Test on iOS device (iOS 15+)"
echo "4. Monitor performance and optimize as needed"
