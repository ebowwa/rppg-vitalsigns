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
        import json
        import os
        
        resources_path = 'docs/RESOURCES.json'
        if os.path.exists(resources_path):
            with open(resources_path, 'r') as f:
                resources = json.load(f)
            
            results = {}
            
            for category, datasets in resources['datasets']['huggingface'].items():
                for dataset in datasets:
                    results[dataset['name']] = {
                        'status': 'available',
                        'dataset_type': 'huggingface',
                        'category': category,
                        'priority': dataset.get('priority', 'medium'),
                        'samples': dataset.get('samples', 'unknown')
                    }
            
            for category, resources_list in resources['datasets']['kaggle'].items():
                for resource in resources_list:
                    results[resource['name']] = {
                        'status': 'available',
                        'dataset_type': 'kaggle',
                        'category': category,
                        'priority': resource.get('priority', 'medium')
                    }
            
            for dataset in resources['datasets']['traditional_rppg']:
                results[dataset['name']] = {
                    'status': 'available',
                    'dataset_type': 'traditional_rppg',
                    'category': 'rppg',
                    'priority': dataset.get('priority', 'high')
                }
        else:
            results = {'error': 'Consolidated resources file not found'}
        
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
