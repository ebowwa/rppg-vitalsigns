#!/usr/bin/env python3
"""
Test script for dataset validation functionality
"""

import sys
import os
sys.path.append('.')

def test_dataset_validation():
    """Test the dataset validation functionality"""
    print("Testing dataset validation...")
    
    try:
        from scripts.data_pipeline_automation import validate_datasets
        
        results = validate_datasets()
        
        print(f"Validation completed successfully!")
        print(f"Found {len(results)} dataset entries.")
        
        for i, (name, result) in enumerate(results.items()):
            if i < 5:  # Show first 5 results
                status = result.get('status', 'unknown')
                dataset_type = result.get('dataset_type', 'unknown')
                print(f"  {name}: {status} ({dataset_type})")
        
        status_counts = {}
        for result in results.values():
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\nStatus summary:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        print("\nDataset validation test passed!")
        return True
        
    except Exception as e:
        print(f"Dataset validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_validation()
    sys.exit(0 if success else 1)
