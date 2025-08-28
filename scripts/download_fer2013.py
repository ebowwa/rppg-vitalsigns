import os
import kaggle
from pathlib import Path

def download_fer2013(data_dir="./data"):
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    fer2013_path = data_path / "fer2013"
    
    if not fer2013_path.exists():
        print("Downloading FER2013 dataset from Kaggle...")
        
        os.environ['KAGGLE_CONFIG_DIR'] = str(Path.home() / '.kaggle')
        
        kaggle.api.dataset_download_files(
            'msambare/fer2013',
            path=str(data_path),
            unzip=True
        )
        
        print(f"FER2013 dataset downloaded to {fer2013_path}")
    else:
        print(f"FER2013 dataset already exists at {fer2013_path}")
    
    return fer2013_path

if __name__ == "__main__":
    download_fer2013()
