"""Medical Dataset Loading and Federated Distribution

Provides unified interface for medical imaging datasets and utilities for
creating federated splits across clients.
"""

import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np

from .mimic_cxr import MIMICCXRDataset, create_mimic_cxr_loader
from .medmnist import MedMNISTDataset, create_medmnist_loader
from .chexpert import CheXpertDataset, create_chexpert_loader

logger = logging.getLogger(__name__)


def get_medical_dataset(
    dataset_name: str,
    root_dir: str,
    split: str = 'train',
    **kwargs
) -> Dataset:
    """Get medical imaging dataset.
    
    Args:
        dataset_name: Dataset name ('mimic_cxr', 'medmnist', 'chexpert')
        root_dir: Path to dataset directory
        split: 'train', 'val', or 'test'
        **kwargs: Additional arguments for specific dataset
    
    Returns:
        Dataset instance
    
    Raises:
        ValueError: If dataset not supported
    """
    dataset_lower = dataset_name.lower()
    
    if dataset_lower in ['mimic_cxr', 'mimic-cxr', 'mimiccxr']:
        return MIMICCXRDataset(root_dir, split=split, **kwargs)
    
    elif dataset_lower in ['medmnist', 'med-mnist']:
        dataset_type = kwargs.pop('dataset_type', 'chest')
        return MedMNISTDataset(
            root_dir,
            dataset_name=dataset_type,
            split=split,
            **kwargs
        )
    
    elif dataset_lower in ['chexpert', 'chex-pert']:
        return CheXpertDataset(root_dir, split=split, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Supported: 'mimic_cxr', 'medmnist', 'chexpert'"
        )


def get_dataset_info(dataset_name: str) -> Dict:
    """Get dataset information.
    
    Args:
        dataset_name: Dataset name
    
    Returns:
        Dict with dataset metadata
    """
    dataset_lower = dataset_name.lower()
    
    if dataset_lower in ['mimic_cxr', 'mimic-cxr']:
        return MIMICCXRDataset.get_info()
    elif dataset_lower in ['medmnist', 'med-mnist']:
        return MedMNISTDataset.get_info()
    elif dataset_lower in ['chexpert', 'chex-pert']:
        return CheXpertDataset.get_info()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_federated_splits(
    dataset: Dataset,
    num_clients: int = 5,
    split_type: str = 'iid',
    alpha: float = 0.1,
    seed: int = 42
) -> List[Subset]:
    """Create federated data splits across clients.
    
    Args:
        dataset: Training dataset
        num_clients: Number of federated clients
        split_type: 'iid' (independent and identically distributed) or 'non_iid'
        alpha: Dirichlet parameter for non-IID distribution (lower = more non-IID)
        seed: Random seed for reproducibility
    
    Returns:
        List of Subset objects, one per client
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_data = len(dataset)
    indices = np.arange(n_data)
    
    if split_type == 'iid':
        # Shuffle and split evenly
        np.random.shuffle(indices)
        split_size = n_data // num_clients
        client_indices = [
            indices[i*split_size:(i+1)*split_size]
            for i in range(num_clients)
        ]
    
    elif split_type == 'non_iid':
        # Use Dirichlet distribution for non-IID split
        try:
            if hasattr(dataset, 'labels') and dataset.labels is not None:
                raw = dataset.labels
                labels = np.array(raw) if not isinstance(raw, np.ndarray) else raw
            else:
                sample = dataset[0]
                label_key = 'label' if 'label' in sample else 'labels'
                raw = [dataset[i][label_key] for i in range(n_data)]
                labels = np.array([l.item() if isinstance(l, torch.Tensor) and l.dim() == 0
                                   else (l.numpy() if isinstance(l, torch.Tensor) else l)
                                   for l in raw])

            if labels.ndim > 1 and labels.shape[1] > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.flatten().astype(int)
        except (TypeError, KeyError):
            logger.warning("Could not extract labels for non-IID split, using random")
            np.random.shuffle(indices)
            split_size = n_data // num_clients
            client_indices = [
                indices[i*split_size:(i+1)*split_size]
                for i in range(num_clients)
            ]
            return [Subset(dataset, idx) for idx in client_indices]
        
        # Dirichlet distribution over classes
        n_classes = len(np.unique(labels))
        label_distribution = np.random.dirichlet(
            [alpha] * n_classes,
            num_clients
        )
        
        client_indices = [[] for _ in range(num_clients)]
        
        # Assign samples to clients based on Dirichlet distribution
        for label in range(n_classes):
            label_indices = np.where(labels == label)[0]
            np.random.shuffle(label_indices)
            
            # Split label indices among clients
            splits = np.split(
                label_indices,
                np.cumsum(
                    (label_distribution[:, label] * len(label_indices)).astype(int)
                )[:-1]
            )
            
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split)
        
        client_indices = [
            np.array(client_indices[i])
            for i in range(num_clients)
        ]
    
    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    
    logger.info(
        f"Created {num_clients} federated data splits ({split_type}): "
        f"Sizes: {[len(idx) for idx in client_indices]}"
    )
    
    return [Subset(dataset, idx) for idx in client_indices]


def create_federated_loaders(
    dataset: Dataset,
    num_clients: int = 5,
    batch_size: int = 32,
    split_type: str = 'iid',
    alpha: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> List[DataLoader]:
    """Create federated DataLoaders.
    
    Args:
        dataset: Training dataset
        num_clients: Number of clients
        batch_size: Batch size
        split_type: 'iid' or 'non_iid'
        alpha: Dirichlet parameter
        num_workers: Number of loading workers
        seed: Random seed
    
    Returns:
        List of DataLoaders for each client
    """
    client_subsets = create_federated_splits(
        dataset,
        num_clients=num_clients,
        split_type=split_type,
        alpha=alpha,
        seed=seed
    )
    
    loaders = [
        DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        for subset in client_subsets
    ]
    
    return loaders


def compute_dataset_statistics(dataset: Dataset) -> Dict:
    """Compute dataset statistics.
    
    Args:
        dataset: Dataset instance
    
    Returns:
        Dict with statistics (mean, std, etc.)
    """
    logger.info("Computing dataset statistics...")
    
    all_images = []
    all_labels = []
    
    for i in range(min(1000, len(dataset))):  # Sample 1000 for efficiency
        batch = dataset[i]
        all_images.append(batch['image'])
        
        if 'label' in batch:
            all_labels.append(batch['label'])
        elif 'labels' in batch:
            all_labels.append(batch['labels'])
    
    all_images = torch.stack(all_images)
    
    stats = {
        'n_samples': len(dataset),
        'image_shape': tuple(all_images.shape),
        'mean': all_images.mean(dim=0).numpy().tolist(),
        'std': all_images.std(dim=0).numpy().tolist(),
        'min': all_images.min().item(),
        'max': all_images.max().item()
    }
    
    if all_labels:
        all_labels = torch.stack(all_labels)
        if all_labels.dim() > 1:
            stats['n_classes'] = all_labels.shape[1]
        else:
            stats['n_classes'] = len(torch.unique(all_labels))
    
    logger.info(f"Statistics: {stats}")
    
    return stats
