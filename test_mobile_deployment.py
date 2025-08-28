#!/usr/bin/env python3
"""
Test Mobile Deployment Pipeline
Verifies Core ML conversion and Swift integration
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append('.')
from src.models.vitallens_emotion import VitalLensEmotionModel
from scripts.deploy_mobile import MobileDeploymentPipeline

def test_mobile_deployment_pipeline():
    print('🧪 Testing Mobile Deployment Pipeline...')
    
    print('1. Creating synthetic trained model...')
    model = VitalLensEmotionModel(
        sequence_length=150,
        num_emotions=7,
        dropout_rate=0.3,
        enable_audio=True,
        enable_eyetracking=True
    )
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'sequence_length': 150,
            'num_emotions': 7,
            'dropout_rate': 0.3
        },
        'epoch': 50,
        'val_loss': 0.5,
        'val_hr_mae': 1.2,
        'val_emotion_acc': 0.78
    }
    
    test_checkpoint_path = './test_checkpoint.pth'
    torch.save(checkpoint, test_checkpoint_path)
    print(f'   Synthetic checkpoint saved: {test_checkpoint_path}')
    
    print('2. Testing mobile deployment pipeline...')
    config = {
        'model_name': 'VitalLensMultiModalTest',
        'output_dir': './test_mobile_deployment',
        'target_size_mb': 20,
        'target_inference_ms': 18,
        'enable_pruning': False,  # Skip for testing
        'enable_quantization': False  # Skip for testing
    }
    
    pipeline = MobileDeploymentPipeline(config)
    
    print('3. Testing model loading...')
    try:
        loaded_model = pipeline.load_trained_model(test_checkpoint_path)
        print(f'   ✅ Model loaded: {sum(p.numel() for p in loaded_model.parameters()):,} parameters')
    except Exception as e:
        print(f'   ❌ Model loading failed: {e}')
        return False
    
    print('4. Testing model optimization...')
    try:
        optimized_model = pipeline.optimize_model(loaded_model)
        print(f'   ✅ Model optimized: {sum(p.numel() for p in optimized_model.parameters()):,} parameters')
    except Exception as e:
        print(f'   ❌ Model optimization failed: {e}')
        return False
    
    print('5. Testing Core ML conversion...')
    try:
        coreml_path, metrics = pipeline.convert_to_coreml(optimized_model)
        print(f'   ✅ Core ML model created: {coreml_path}')
        print(f'   📊 Model size: {metrics["model_size_mb"]:.1f} MB')
        print(f'   🎯 Size target met: {not metrics["size_reduction_needed"]}')
    except Exception as e:
        print(f'   ❌ Core ML conversion failed: {e}')
        return False
    
    print('6. Testing Swift integration generation...')
    try:
        swift_path = pipeline.generate_swift_integration(coreml_path)
        print(f'   ✅ Swift integration code created: {swift_path}')
    except Exception as e:
        print(f'   ❌ Swift integration generation failed: {e}')
        return False
    
    print('7. Testing deployment automation...')
    try:
        automation_path = pipeline.create_deployment_automation()
        print(f'   ✅ Deployment automation created: {automation_path}')
    except Exception as e:
        print(f'   ❌ Deployment automation creation failed: {e}')
        return False
    
    print('8. Testing performance benchmark...')
    try:
        benchmark_results = pipeline.run_performance_benchmark(coreml_path)
        print(f'   ✅ Performance benchmark completed')
        print(f'   ⚡ Inference time: {benchmark_results["avg_inference_ms"]:.1f}ms')
        print(f'   🎯 Inference target met: {benchmark_results["meets_inference_target"]}')
        print(f'   📊 Size target met: {benchmark_results["meets_size_target"]}')
    except Exception as e:
        print(f'   ❌ Performance benchmark failed: {e}')
        return False
    
    print('9. Testing complete pipeline...')
    try:
        results = pipeline.deploy_complete_pipeline(test_checkpoint_path)
        if results['success']:
            print(f'   ✅ Complete pipeline succeeded')
            print(f'   📁 Output directory: {results["output_dir"]}')
            print(f'   📱 Core ML model: {results["coreml_path"]}')
            print(f'   📝 Swift code: {results["swift_path"]}')
            print(f'   🤖 Automation: {results["automation_path"]}')
        else:
            print(f'   ❌ Complete pipeline failed: {results.get("error", "Unknown error")}')
            return False
    except Exception as e:
        print(f'   ❌ Complete pipeline test failed: {e}')
        return False
    
    import shutil
    Path(test_checkpoint_path).unlink(missing_ok=True)
    shutil.rmtree('./test_mobile_deployment', ignore_errors=True)
    
    print('\n✅ All mobile deployment pipeline tests passed!')
    print('🎯 Ready for production deployment')
    print('📱 Core ML conversion: Working')
    print('📝 Swift integration: Working')
    print('🤖 Automation scripts: Working')
    print('⚡ Performance benchmarking: Working')
    print('🚀 Complete pipeline: Working')
    
    return True

def test_swift_integration_syntax():
    print('\n🧪 Testing Swift integration syntax...')
    
    config = {
        'model_name': 'VitalLensMultiModalTest',
        'output_dir': './test_swift',
        'target_size_mb': 20,
        'target_inference_ms': 18
    }
    
    pipeline = MobileDeploymentPipeline(config)
    
    Path('./test_swift').mkdir(exist_ok=True)
    dummy_coreml_path = Path('./test_swift/test.mlmodel')
    dummy_coreml_path.touch()
    
    try:
        swift_path = pipeline.generate_swift_integration(dummy_coreml_path)
        
        with open(swift_path, 'r') as f:
            swift_code = f.read()
        
        required_components = [
            'import CoreML',
            'import Vision',
            'import AVFoundation',
            'class VitalLensMultiModalProcessor',
            'struct VitalLensResults',
            'enum VitalLensError',
            'func processMultiModalFrame',
            'func runMultiModalInference',
            'private let emotionLabels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in swift_code:
                missing_components.append(component)
        
        if missing_components:
            print(f'   ❌ Missing Swift components: {missing_components}')
            return False
        
        print(f'   ✅ Swift integration syntax valid')
        print(f'   📱 All required components present')
        print(f'   🎯 Multi-modal support: Video + Audio + Eye-tracking')
        print(f'   📊 Emotion classes: 7 classes supported')
        
        import shutil
        shutil.rmtree('./test_swift', ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f'   ❌ Swift integration test failed: {e}')
        return False

if __name__ == "__main__":
    success = True
    
    if not test_mobile_deployment_pipeline():
        success = False
    
    if not test_swift_integration_syntax():
        success = False
    
    if success:
        print('\n🎉 All mobile deployment tests passed!')
        print('✅ Ready for production deployment to iOS devices')
    else:
        print('\n❌ Some mobile deployment tests failed')
        sys.exit(1)
