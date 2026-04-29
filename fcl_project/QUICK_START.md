# Quick Start Guide - Medium-Term Implementation

## 🚀 Get Started in 5 Minutes

### What You Now Have

✅ **Medical Imaging Datasets**
- MIMIC-CXR (377K images, real hospital data)
- MedMNIST (7 datasets, 10-240K images each)
- CheXpert (224K images, Stanford)

✅ **Privacy Auditing**
- Membership Inference Attacks (MIA)
- Quantified privacy metrics (AUC, TPR, FPR)
- Privacy level classification

✅ **Research Visualizations**
- ROC curves, confusion matrices
- Privacy-utility tradeoffs
- 300 DPI publication-grade plots

---

## 📦 Installation

```bash
# Navigate to project
cd /home/bs01233/Documents/FL/fcl_project

# Install core dependencies
pip install torch torchvision scikit-learn matplotlib seaborn pandas numpy

# Optional: Privacy/Federated components
pip install opacus flower  # For advanced features
```

---

## 💻 Quick Examples

### 1. Load Medical Dataset

```python
from code.datasets import get_medical_dataset

# Load MIMIC-CXR
dataset = get_medical_dataset(
    'mimic_cxr',
    'data/mimic_cxr',
    split='train'
)

# Load MedMNIST (auto-downloads)
dataset = get_medical_dataset(
    'medmnist',
    'data/medmnist',
    dataset_type='chest'  # or 'path', 'derma', etc.
)

# Load CheXpert
dataset = get_medical_dataset(
    'chexpert',
    'data/chexpert',
    split='train'
)

# Get sample
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Labels shape: {sample['labels'].shape}")
```

### 2. Create Federated Splits

```python
from code.datasets import create_federated_loaders

# Create federated data for 5 clients
loaders = create_federated_loaders(
    dataset,
    num_clients=5,
    batch_size=32,
    split_type='non_iid',  # Realistic heterogeneous data
    alpha=0.1  # Lower = more non-IID
)

# Access per-client data
for client_id, loader in enumerate(loaders):
    print(f"Client {client_id}: {len(loader.dataset)} samples")
    
    for batch in loader:
        images = batch['image']
        labels = batch['labels']
        break
```

### 3. Privacy Auditing (MIA)

```python
from code.privacy_audit import (
    MembershipInferenceAttack,
    MIAConfig,
    evaluate_differential_privacy_protection
)
import torch.nn as nn

# Configure attack
mia_config = MIAConfig(
    n_shadow_models=5,
    shadow_model_epochs=20,
    device='cuda'
)

# Create model constructor
def model_constructor():
    # Return new model instance
    return YourModel()

# Run privacy audit
mia = MembershipInferenceAttack(model_constructor, mia_config)
mia.train_shadow_models(dataset, batch_size=32)
results = mia.evaluate_attack()

print(f"AUC: {results['auc']:.4f}")
print(f"TPR @ FPR=0.5: {results['tpr_at_50_fpr']:.4f}")
print(f"Privacy Level: {results['privacy_level']}")
```

### 4. Generate Publication Plots

```python
from code.visualization import PublicationVisualizer
import numpy as np

# Set publication style
PublicationVisualizer.set_paper_style()

# Plot ROC curves
results = {
    'Model A': {'fpr': fpr_a, 'tpr': tpr_a, 'auc': 0.85},
    'Model B': {'fpr': fpr_b, 'tpr': tpr_b, 'auc': 0.82}
}

fig, ax = PublicationVisualizer.plot_roc_curves(
    results,
    title="Model Comparison",
    save_path="results/roc_curves.pdf"  # 300 DPI!
)

# Confusion matrix
PublicationVisualizer.plot_confusion_matrix(
    y_true=test_labels,
    y_pred=predictions,
    class_names=['Normal', 'Abnormal'],
    save_path="results/confusion.pdf"
)

# Privacy-utility tradeoff
PublicationVisualizer.plot_privacy_utility_tradeoff(
    epsilon_values=[0.1, 0.5, 1.0, 5.0],
    accuracy_values=[[0.75, 0.80, 0.82, 0.85]],
    attack_success=[[80, 70, 55, 35]],
    model_names=['FTT+DP'],
    save_path="results/privacy_utility.pdf"
)
```

### 5. Run Full Experiment

```python
from scripts.run_medical_imaging_experiments import (
    MedicalImagingExperiment,
    ExperimentConfig
)

# Configure experiment
config = ExperimentConfig(
    experiment_name="mimic_cxr_baseline",
    dataset_name='mimic_cxr',
    dataset_path='data/mimic_cxr',
    num_epochs=20,
    batch_size=32,
    learning_rate=0.001,
    device='cuda',
    output_dir='results'
)

# Run experiment
experiment = MedicalImagingExperiment(config)
results = experiment.run()

print(f"Test Accuracy: {results['test_accuracy']:.4f}")

# Results saved to: results/mimic_cxr_baseline/
#   - experiment.log
#   - results.json
#   - best_model.pth
```

---

## 📊 File Organization

```
code/datasets/              Medical imaging datasets
  ├── mimic_cxr.py        MIMIC-CXR (233 lines)
  ├── medmnist.py         MedMNIST (245 lines)
  ├── chexpert.py         CheXpert (241 lines)
  └── loader.py           Federated utilities (270 lines)

code/privacy_audit.py       Membership Inference Attacks (400 lines)
code/visualization.py       Publication plots (477 lines)
scripts/run_medical_imaging_experiments.py  (418 lines)
```

---

## 🎯 Common Tasks

### Download Datasets

**MedMNIST** (Auto-download):
```python
from code.datasets import MedMNISTDataset
dataset = MedMNISTDataset('data/medmnist', dataset_name='chest')
# Auto-downloads from Zenodo
```

**MIMIC-CXR** (Manual):
1. Register at https://physionet.org/
2. Get credentialed access
3. Download from https://physionet.org/content/mimic-cxr/
4. Extract to `data/mimic_cxr/`

**CheXpert** (Manual):
1. Register at https://stanfordmlgroup.github.io/competitions/chexpert/
2. Download metadata and images
3. Extract to `data/chexpert/`

### Check Dataset Info

```python
from code.datasets import get_dataset_info

info = get_dataset_info('mimic_cxr')
print(f"Images: {info['total_images']}")
print(f"Patients: {info['unique_patients']}")
print(f"Findings: {info['findings']}")
```

### Compute Dataset Statistics

```python
from code.datasets.loader import compute_dataset_statistics

stats = compute_dataset_statistics(dataset)
print(f"Mean: {stats['mean']}")
print(f"Std: {stats['std']}")
print(f"Classes: {stats['n_classes']}")
```

---

## 📈 Expected Results

### Model Accuracy
- **MIMIC-CXR**: 75-85% (14-label classification)
- **ChestMNIST**: 85-92% (binary classification)
- **DermaMNIST**: 80-90% (7-class classification)

### Privacy Metrics
- **Strong DP (ε=0.1)**: AUC ~0.50-0.55 (good privacy)
- **Moderate DP (ε=1.0)**: AUC ~0.55-0.60
- **Weak DP (ε=5.0)**: AUC ~0.60-0.70 (poor privacy)

### Federated Learning
- **Non-IID (α=0.1)**: Slower convergence, ~20-30 rounds
- **IID**: Fast convergence, ~10-15 rounds

---

## 🔍 Debugging Tips

### Check Dataset Loading
```python
from code.datasets import get_medical_dataset

try:
    dataset = get_medical_dataset('mimic_cxr', 'data/mimic_cxr')
    print(f"✅ Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check data path and dataset availability")
```

### Verify Privacy Metrics
```python
results = mia.evaluate_attack()

if results['auc'] > 0.6:
    print("⚠️ Privacy is weak - DP protection is insufficient")
elif results['auc'] > 0.55:
    print("⚠️ Privacy is moderate")
else:
    print("✅ Privacy is strong - Good DP protection")
```

### Check Visualization Output
```python
import os

if os.path.exists('results/roc_curves.pdf'):
    print("✅ Visualization saved (300 DPI)")
else:
    print("❌ Visualization not found")
```

---

## 📚 Full Documentation

For detailed information, see:
- **MEDIUM_TERM_IMPLEMENTATION.md** - Comprehensive guide
- **PROJECT_OVERVIEW.md** - Complete project status
- **Docstrings** - In-code API reference

---

## 🎓 What You Can Research

1. **Federated Medical Learning**
   - Privacy-preserving training on hospital data
   - Multi-domain federated learning
   - Heterogeneous client scenarios

2. **Privacy-Utility Tradeoffs**
   - Impact of DP noise on accuracy
   - MIA attack success rates
   - DP-SGD effectiveness

3. **Medical Imaging AI**
   - Multi-modal diagnosis
   - Multi-label classification
   - Uncertainty handling

4. **Federated Learning**
   - Communication efficiency
   - Model aggregation strategies
   - Convergence analysis

---

## ✅ Checklist Before Running

- [ ] PyTorch installed
- [ ] Dependencies installed (`pip install ...`)
- [ ] Dataset path configured
- [ ] Output directory exists
- [ ] GPU available (optional but recommended)

---

## 🚀 Start Here!

```bash
# 1. Navigate to project
cd /home/bs01233/Documents/FL/fcl_project

# 2. Install dependencies (if not done)
pip install torch torchvision scikit-learn matplotlib seaborn pandas

# 3. Try loading a dataset
python -c "
from code.datasets import MedMNISTDataset
d = MedMNISTDataset('data', dataset_name='chest')
print(f'✅ Dataset loaded: {len(d)} samples')
"

# 4. Run experiment
python scripts/run_medical_imaging_experiments.py
```

---

**Ready to go!** 🚀

For any questions, refer to the comprehensive documentation files included in the project.
