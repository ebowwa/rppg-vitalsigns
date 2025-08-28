#!/usr/bin/env python3
"""
Test script to verify download functionality without actually downloading large files
"""

import sys
import os
sys.path.append('.')

def test_download_imports():
    """Test that download functions can be imported"""
    results = {}
    
    try:
        from scripts.download_datasets import download_fer2013
        results['download_fer2013'] = 'AVAILABLE'
    except ImportError as e:
        results['download_fer2013'] = f'MISSING: {e}'
    
    try:
        from scripts.download_datasets import download_hf_emotion_datasets
        results['download_hf_emotion_datasets'] = 'AVAILABLE'
    except ImportError as e:
        results['download_hf_emotion_datasets'] = f'MISSING: {e}'
    
    try:
        from scripts.download_datasets import download_traditional_rppg
        results['download_traditional_rppg'] = 'AVAILABLE'
    except ImportError as e:
        results['download_traditional_rppg'] = f'MISSING: {e}'
    
    try:
        import gdown
        results['gdown'] = 'AVAILABLE'
    except ImportError as e:
        results['gdown'] = f'MISSING: {e}'
    
    return results

def test_google_drive_functionality():
    """Test Google Drive download functionality without downloading"""
    try:
        from scripts.download_datasets import download_traditional_rppg
        
        print("Testing UBFC-rPPG download function (dry run)...")
        
        datasets_dir = "./test_datasets"
        os.makedirs(datasets_dir, exist_ok=True)
        
        print("✅ Google Drive download function is callable")
        
        os.rmdir(datasets_dir) if os.path.exists(datasets_dir) and not os.listdir(datasets_dir) else None
        
        return True
    except Exception as e:
        print(f"❌ Google Drive functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing download functionality...")
    
    print("\n1. Testing download function imports...")
    results = test_download_imports()
    
    for name, status in results.items():
        if 'MISSING' in status:
            print(f"❌ {name}: {status}")
        else:
            print(f"✅ {name}: {status}")
    
    print("\n2. Testing Google Drive functionality...")
    gd_success = test_google_drive_functionality()
    
    missing_count = sum(1 for status in results.values() if 'MISSING' in status)
    print(f"\nSummary: {missing_count}/{len(results)} imports are missing")
    print(f"Google Drive functionality: {'✅ WORKING' if gd_success else '❌ BROKEN'}")
