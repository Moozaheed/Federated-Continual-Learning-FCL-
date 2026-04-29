"""MedMNIST Dataset Loaders

Collection of medical imaging datasets with consistent interface.
Source: https://medmnist.com/
Papers: Yang et al. "MedMNIST Dataset" Nature Scientific Data 2021

Includes:
- PathMNIST: Colorectal polyp classification (9 classes)
- ChestMNIST: Chest X-ray classification (11 classes)
- DermaMNIST: Skin lesion classification (7 classes)
- RetinaMNIST: Retinal fundus photo classification (5 classes)
- BloodMNIST: Blood cell morphology (8 classes)
- TissueMNIST: Histopathology tissue (8 classes)
- OrganMNIST: CT organ classification (11 classes)
"""

import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import h5py

logger = logging.getLogger(__name__)


class MedMNISTDataset(Dataset):
    """MedMNIST dataset loader.
    
    Args:
        root_dir: Path to dataset
        dataset_name: One of 'path', 'chest', 'derma', 'retina', 'blood', 'tissue', 'organ'
        split: 'train', 'val', or 'test'
        image_size: Resize to (image_size, image_size)
        normalize: Apply normalization
        transform: Additional transforms
    """
    
    DATASET_INFO = {
        'path': {
            'name': 'PathMNIST',
            'n_classes': 9,
            'modality': 'Histopathology',
            'n_images': 89996,
            'url': 'https://zenodo.org/record/5208230/files/pathmnist.npz'
        },
        'chest': {
            'name': 'ChestMNIST',
            'n_classes': 11,
            'modality': 'Chest X-ray',
            'n_images': 112009,
            'url': 'https://zenodo.org/record/5208230/files/chestmnist.npz'
        },
        'derma': {
            'name': 'DermaMNIST',
            'n_classes': 7,
            'modality': 'Dermatology',
            'n_images': 10015,
            'url': 'https://zenodo.org/record/5208230/files/dermamnist.npz'
        },
        'retina': {
            'name': 'RetinaMNIST',
            'n_classes': 5,
            'modality': 'Retinal Fundus',
            'n_images': 9592,
            'url': 'https://zenodo.org/record/5208230/files/retinamnist.npz'
        },
        'blood': {
            'name': 'BloodMNIST',
            'n_classes': 8,
            'modality': 'Blood Cell',
            'n_images': 17092,
            'url': 'https://zenodo.org/record/5208230/files/bloodmnist.npz'
        },
        'tissue': {
            'name': 'TissueMNIST',
            'n_classes': 8,
            'modality': 'Histology',
            'n_images': 236386,
            'url': 'https://zenodo.org/record/5208230/files/tissuemnist.npz'
        },
        'organ': {
            'name': 'OrganMNIST',
            'n_classes': 11,
            'modality': 'CT Organs',
            'n_images': 58850,
            'url': 'https://zenodo.org/record/5208230/files/organmnist.npz'
        }
    }
    
    def __init__(
        self,
        root_dir: str,
        dataset_name: str = 'chest',
        split: str = 'train',
        image_size: int = 224,
        normalize: bool = True,
        transform: Optional[transforms.Compose] = None
    ):
        """Initialize MedMNIST dataset."""
        assert dataset_name in self.DATASET_INFO, \
            f"Unknown dataset: {dataset_name}. Choose from {list(self.DATASET_INFO.keys())}"
        
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.transform = transform or self._get_default_transform(normalize)
        self.info = self.DATASET_INFO[dataset_name]
        
        # Load data
        self.images, self.labels = self._load_data()
        
        logger.info(
            f"Loaded {self.info['name']} {split} split: {len(self.images)} images"
        )
    
    def _get_default_transform(self, normalize: bool) -> transforms.Compose:
        """Get default preprocessing."""
        transform_list = [
            transforms.ToTensor(),
        ]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.5] * 3,
                    std=[0.5] * 3
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from .npz file."""
        data_file = self.root_dir / f'{self.dataset_name}mnist.npz'
        
        if not data_file.exists():
            logger.warning(
                f"Data file not found: {data_file}. "
                f"Download from {self.info['url']}"
            )
            return np.array([]), np.array([])
        
        data = np.load(data_file)
        
        # Get split data
        split_key = f'{self.split}_images'
        label_key = f'{self.split}_labels'
        
        if split_key in data and label_key in data:
            images = data[split_key]
            labels = data[label_key].squeeze()
            return images, labels
        
        return np.array([]), np.array([])
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get image and label.
        
        Returns:
            Dict with keys:
                - 'image': Tensor (C, 28, 28) or resized
                - 'label': Class label (0 to n_classes-1)
        """
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image
        if image.shape[-1] == 1:  # Grayscale
            image = Image.fromarray(image[..., 0], mode='L')
            # Convert to RGB for consistency
            image = image.convert('RGB')
        else:  # RGB
            image = Image.fromarray(image, mode='RGB')
        
        # Resize if needed
        if self.image_size != 28:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Apply transforms
        image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'dataset': self.dataset_name
        }
    
    @classmethod
    def get_info(cls, dataset_name: str = None) -> Dict:
        """Get dataset information."""
        if dataset_name is None:
            return {name: info for name, info in cls.DATASET_INFO.items()}
        else:
            return cls.DATASET_INFO.get(dataset_name, {})


def create_medmnist_loader(
    root_dir: str,
    dataset_name: str = 'chest',
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    image_size: int = 224
) -> DataLoader:
    """Create MedMNIST DataLoader.
    
    Args:
        root_dir: Path to dataset directory
        dataset_name: Dataset name
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of loading workers
        image_size: Resize images to
    
    Returns:
        DataLoader for MedMNIST
    """
    dataset = MedMNISTDataset(
        root_dir=root_dir,
        dataset_name=dataset_name,
        split=split,
        image_size=image_size
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader
