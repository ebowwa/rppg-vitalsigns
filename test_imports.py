#!/usr/bin/env python3
"""
Test script to check for missing imports after cleanup
"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Test critical imports that might be missing"""
    results = {}
    
    try:
        from src.models.vitallens_emotion import VitalLensEmotionModel
        results['VitalLensEmotionModel'] = 'AVAILABLE'
    except ImportError as e:
        results['VitalLensEmotionModel'] = f'MISSING: {e}'
    
    try:
        from src.models.vitallens_model import VitalLensModel
        results['VitalLensModel'] = 'AVAILABLE'
    except ImportError as e:
        results['VitalLensModel'] = f'MISSING: {e}'
    
    try:
        from src.data.dataset import RPPGEmotionDataset
        results['RPPGEmotionDataset'] = 'AVAILABLE'
    except ImportError as e:
        results['RPPGEmotionDataset'] = f'MISSING: {e}'
    
    try:
        from src.processing.face_detection import FaceDetectionProcessor
        results['FaceDetectionProcessor'] = 'AVAILABLE'
    except ImportError as e:
        results['FaceDetectionProcessor'] = f'MISSING: {e}'
    
    try:
        from src.processing.signal_quality import SignalQualityAssessment
        results['SignalQualityAssessment'] = 'AVAILABLE'
    except ImportError as e:
        results['SignalQualityAssessment'] = f'MISSING: {e}'
    
    try:
        from src.training.trainer import VitalLensTrainer
        results['VitalLensTrainer'] = 'AVAILABLE'
    except ImportError as e:
        results['VitalLensTrainer'] = f'MISSING: {e}'
    
    return results

if __name__ == "__main__":
    print("Testing imports after cleanup...")
    results = test_imports()
    
    for name, status in results.items():
        if 'MISSING' in status:
            print(f"❌ {name}: {status}")
        else:
            print(f"✅ {name}: {status}")
    
    missing_count = sum(1 for status in results.values() if 'MISSING' in status)
    print(f"\nSummary: {missing_count}/{len(results)} imports are missing")
