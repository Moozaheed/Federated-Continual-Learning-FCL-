"""Medical Imaging Datasets Module

Provides loaders and preprocessing for:
- MIMIC-CXR (chest X-ray)
- MedMNIST (diverse medical imaging)
- CheXpert (chest radiographs)
"""

from .mimic_cxr import MIMICCXRDataset
from .medmnist import MedMNISTDataset
from .chexpert import CheXpertDataset
from .loader import (
    get_medical_dataset,
    create_federated_splits,
    get_dataset_info
)

__all__ = [
    'MIMICCXRDataset',
    'MedMNISTDataset',
    'CheXpertDataset',
    'get_medical_dataset',
    'create_federated_splits',
    'get_dataset_info'
]
