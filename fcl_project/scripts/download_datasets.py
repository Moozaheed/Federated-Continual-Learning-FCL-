import os
import requests
from tqdm import tqdm
from pathlib import Path
import time

# Latest MedMNIST Zenodo Record: 10519652
BASE_URL = "https://zenodo.org/records/10519652/files/"

DATASET_INFO = {
    'path': {
        'name': 'PathMNIST',
        'file': 'pathmnist.npz',
        'min_size': 190 * 1024 * 1024
    },
    'chest': {
        'name': 'ChestMNIST',
        'file': 'chestmnist.npz',
        'min_size': 75 * 1024 * 1024
    },
    'derma': {
        'name': 'DermaMNIST',
        'file': 'dermamnist.npz',
        'min_size': 18 * 1024 * 1024
    },
    'retina': {
        'name': 'RetinaMNIST',
        'file': 'retinamnist.npz',
        'min_size': 3 * 1024 * 1024
    },
    'blood': {
        'name': 'BloodMNIST',
        'file': 'bloodmnist.npz',
        'min_size': 33 * 1024 * 1024
    },
    'tissue': {
        'name': 'TissueMNIST',
        'file': 'tissuemnist.npz',
        'min_size': 115 * 1024 * 1024
    },
    'organ': {
        'name': 'OrganMNIST',
        'file': 'organamnist.npz', # Mapping Axial to general organmnist
        'min_size': 30 * 1024 * 1024
    }
}

def download_file(url, dest_path, retries=3):
    for i in range(retries):
        try:
            # Add ?download=1 to ensure direct download
            download_url = f"{url}?download=1"
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)
            return True
        except Exception as e:
            print(f"\nAttempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(5)
            else:
                return False

def main():
    root_dir = Path('fcl_project/data/medmnist')
    root_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading MedMNIST datasets to {root_dir}...")
    
    for key, info in DATASET_INFO.items():
        # The project expects names like 'pathmnist.npz', 'organmnist.npz'
        dest_path = root_dir / f"{key}mnist.npz"
        
        # Check if exists and size is reasonable
        if dest_path.exists():
            size = dest_path.stat().st_size
            if size >= info['min_size']:
                print(f"✅ {info['name']} already exists and size looks correct ({size/1024/1024:.1f} MB).")
                continue
            else:
                print(f"⚠️ {info['name']} exists but is too small ({size/1024/1024:.2f} MB). Re-downloading...")
                dest_path.unlink()
            
        url = BASE_URL + info['file']
        print(f"Downloading {info['name']} from {url}...")
        if download_file(url, dest_path):
            print(f"✅ Successfully downloaded {info['name']}")
        else:
            print(f"❌ Failed to download {info['name']} after multiple attempts.")

if __name__ == "__main__":
    main()
