"""CheXpert Dataset Loader

Large chest radiograph dataset from Stanford University.
Source: https://stanfordmlgroup.github.io/competitions/chexpert/
Papers: Rajpurkar et al. "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels..." AAAI 2019

Dataset Info:
- 224,316 chest radiograph images
- 65,368 unique patients
- 5 different view types (PA, AP, LATERAL, LL, RL)
- 14 observations (diseases/findings)
- Uncertainty labels (-1 for uncertain, 0 for negative, 1 for positive)
- Front/Lateral/other views
"""

import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


class CheXpertDataset(Dataset):
    """CheXpert chest radiograph dataset.
    
    Args:
        root_dir: Path to CheXpert dataset root
        split: 'train', 'val', or 'test'
        observations: List of observations to use
        uncertainty_handling: 'ignore', 'positive', or 'negative' for uncertain labels
        image_size: Resize to (image_size, image_size)
        normalize: Apply ImageNet normalization
        transform: Additional transforms
    """
    
    OBSERVATIONS = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Pleural Effusion',
        'Pneumonia',
        'Pneumothorax',
        'Support Devices',
        'Enlarged Cardiomediastinum',
        'Fracture',
        'Lung Lesion',
        'Lung Opacity',
        'No Finding',
        'Pleural Other'
    ]
    
    VIEWS = ['PA', 'AP', 'LATERAL']
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        observations: Optional[List[str]] = None,
        uncertainty_handling: str = 'positive',
        view: Optional[str] = None,
        image_size: int = 224,
        normalize: bool = True,
        transform: Optional[transforms.Compose] = None
    ):
        """Initialize CheXpert dataset."""
        self.root_dir = Path(root_dir)
        self.split = split
        self.observations = observations or self.OBSERVATIONS
        self.uncertainty_handling = uncertainty_handling
        self.view = view
        self.image_size = image_size
        self.transform = transform or self._get_default_transform(normalize)
        
        # Load data
        self.data = self._load_csv()
        
        logger.info(
            f"Loaded CheXpert {split} split: {len(self.data)} images"
        )
    
    def _get_default_transform(self, normalize: bool) -> transforms.Compose:
        """Get default preprocessing pipeline."""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _load_csv(self) -> pd.DataFrame:
        """Load CheXpert CSV metadata."""
        csv_file = self.root_dir / f'{self.split}.csv'
        
        if not csv_file.exists():
            logger.warning(
                f"CSV file not found: {csv_file}. "
                "Download from https://stanfordmlgroup.github.io/competitions/chexpert/"
            )
            return pd.DataFrame()
        
        data = pd.read_csv(csv_file)
        
        # Filter by view if specified
        if self.view:
            data = data[data['Frontal/Lateral'] == self.view]
        
        # Remove rows with missing images
        data = data[data['Path'].apply(lambda x: (self.root_dir / x).exists())]
        
        return data.reset_index(drop=True)
    
    def _process_labels(self, row: pd.Series) -> torch.Tensor:
        """Process labels from CSV row.
        
        Handling:
        - -1 (uncertain): Based on uncertainty_handling strategy
        - 0 (negative): 0
        - 1 (positive): 1
        """
        labels = []
        for obs in self.observations:
            if obs in row and not pd.isna(row[obs]):
                value = int(row[obs])
                
                if value == -1:  # Uncertain
                    if self.uncertainty_handling == 'ignore':
                        value = 0  # Treat as negative
                    elif self.uncertainty_handling == 'positive':
                        value = 1  # Treat as positive
                    else:  # 'negative'
                        value = 0
                
                labels.append(max(0, value))  # Ensure 0 or 1
            else:
                labels.append(0)
        
        return torch.tensor(labels, dtype=torch.float)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get image and labels.
        
        Returns:
            Dict with keys:
                - 'image': Preprocessed tensor
                - 'labels': Multi-hot label vector
                - 'path': Image file path
        """
        row = self.data.iloc[idx]
        
        # Load image
        image_path = self.root_dir / row['Path']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Process labels
        labels = self._process_labels(row)
        
        return {
            'image': image,
            'labels': labels,
            'path': str(image_path)
        }
    
    @classmethod
    def get_info(cls) -> Dict:
        """Get dataset information."""
        return {
            'name': 'CheXpert',
            'total_images': 224316,
            'unique_patients': 65368,
            'observations': cls.OBSERVATIONS,
            'n_observations': len(cls.OBSERVATIONS),
            'views': cls.VIEWS,
            'multi_label': True,
            'modality': 'Chest Radiograph',
            'source': 'https://stanfordmlgroup.github.io/competitions/chexpert/',
            'paper': 'Rajpurkar et al., AAAI 2019',
            'image_size_avg': (1024, 1024),
            'recommended_resize': (224, 224)
        }


def create_chexpert_loader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    image_size: int = 224,
    view: Optional[str] = None
) -> DataLoader:
    """Create CheXpert DataLoader.
    
    Args:
        root_dir: Path to CheXpert dataset
        split: 'train' or 'val'
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of loading workers
        image_size: Image resize size
        view: Optional view filter ('PA', 'AP', 'LATERAL')
    
    Returns:
        DataLoader for CheXpert dataset
    """
    dataset = CheXpertDataset(
        root_dir=root_dir,
        split=split,
        view=view,
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
