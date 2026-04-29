"""MIMIC-CXR Dataset Loader

Chest X-ray dataset from Massachusetts Institute of Technology (MIT).
Source: https://physionet.org/content/mimic-cxr/
Papers: Wang et al. "CheXpert: A Large Chest X-ray Dataset with Uncertainty Labels..."

Dataset Info:
- 377,110 images from 65,379 unique patients
- Multiple views (PA, AP, LATERAL, etc.)
- Multiple labels per image (14+ findings)
- Common findings: Cardiomegaly, Edema, Pleural Effusion, etc.
- Requires registration to download
"""

import os
import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json

logger = logging.getLogger(__name__)


class MIMICCXRDataset(Dataset):
    """MIMIC-CXR chest X-ray dataset.
    
    Args:
        root_dir: Path to MIMIC-CXR dataset root
        split: 'train', 'val', or 'test'
        findings: List of findings to include (default: all 14)
        image_size: Resize images to (image_size, image_size)
        normalize: Apply ImageNet normalization
        transform: Additional torchvision transforms
    """
    
    # 14 common thoracic findings in CXR
    FINDINGS = [
        'Cardiomegaly',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices',
        'No Finding',
        'Enlarged Cardiomediastinum',
        'Lung Lesion',
        'Lung Opacity'
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        findings: Optional[List[str]] = None,
        image_size: int = 224,
        normalize: bool = True,
        transform: Optional[transforms.Compose] = None
    ):
        """Initialize MIMIC-CXR dataset."""
        self.root_dir = Path(root_dir)
        self.split = split
        self.findings = findings or self.FINDINGS
        self.image_size = image_size
        self.transform = transform
        
        # Create transform pipeline
        if transform is None:
            self.transform = self._get_default_transform(normalize)
        
        # Load metadata
        self.data = self._load_metadata()
        
        logger.info(
            f"Loaded MIMIC-CXR {split} split: {len(self.data)} images, "
            f"{len(set(d['patient_id'] for d in self.data))} unique patients"
        )
    
    def _get_default_transform(self, normalize: bool) -> transforms.Compose:
        """Get default preprocessing pipeline."""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]
        
        if normalize:
            # ImageNet normalization
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _load_metadata(self) -> List[Dict]:
        """Load image metadata and labels."""
        metadata_file = self.root_dir / f"{self.split}_metadata.json"
        
        if not metadata_file.exists():
            logger.warning(
                f"Metadata file not found: {metadata_file}. "
                "Return empty dataset. Download from https://physionet.org/content/mimic-cxr/"
            )
            return []
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        # Filter findings
        filtered_data = []
        for item in data:
            if self.split in item.get('split', ''):
                filtered_data.append(item)
        
        return filtered_data
    
    def _parse_labels(self, label_dict: Dict) -> torch.Tensor:
        """Parse label dictionary to tensor.
        
        Labels: -1 (uncertain), 0 (negative), 1 (positive)
        Converts to: 0 (negative/uncertain), 1 (positive)
        """
        labels = []
        for finding in self.findings:
            if finding in label_dict:
                value = label_dict[finding]
                # Convert: -1,0 -> 0 (negative), 1 -> 1 (positive)
                label = 1 if value == 1 else 0
            else:
                label = 0  # Default to negative
            labels.append(label)
        
        return torch.tensor(labels, dtype=torch.float)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get image and labels.
        
        Returns:
            Dict with keys:
                - 'image': Preprocessed tensor (3, 224, 224)
                - 'labels': Multi-hot label vector (14,)
                - 'patient_id': Patient identifier
                - 'study_id': Study identifier
                - 'findings': List of positive findings
        """
        item = self.data[idx]
        
        # Load image
        image_path = self.root_dir / item['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Parse labels
        labels = self._parse_labels(item.get('labels', {}))
        
        return {
            'image': image,
            'labels': labels,
            'patient_id': item.get('patient_id'),
            'study_id': item.get('study_id'),
            'findings': item.get('findings', [])
        }
    
    @classmethod
    def get_info(cls) -> Dict:
        """Get dataset information."""
        return {
            'name': 'MIMIC-CXR',
            'total_images': 377110,
            'unique_patients': 65379,
            'findings': cls.FINDINGS,
            'n_findings': len(cls.FINDINGS),
            'multi_label': True,
            'modality': 'Chest X-ray',
            'source': 'https://physionet.org/content/mimic-cxr/',
            'paper': 'Wang et al., Radiology 2017',
            'license': 'MIMIC-IV Clinical Database v2.2 (PhysioNet Credentialed User)',
            'image_size_avg': (2320, 2828),  # Original size
            'recommended_resize': (224, 224)
        }


def create_mimic_cxr_loader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    image_size: int = 224
) -> DataLoader:
    """Create MIMIC-CXR DataLoader.
    
    Args:
        root_dir: Path to MIMIC-CXR dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of loading workers
        image_size: Image resize size
    
    Returns:
        DataLoader for MIMIC-CXR dataset
    """
    dataset = MIMICCXRDataset(
        root_dir=root_dir,
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
