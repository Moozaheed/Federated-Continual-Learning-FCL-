# Medium-Term Implementation (5-8 Weeks): Science & Security Phase

## Overview

This phase focuses on **publication-ready results** using real-world medical imaging datasets and rigorous privacy auditing through Membership Inference Attacks (MIA). The goal is to produce research-grade experimental results suitable for publication in top-tier venues.

---

## 1. Medical Imaging Datasets

### 1.1 MIMIC-CXR Dataset

**File**: `code/datasets/mimic_cxr.py`

**Features**:
- 377,110 chest X-ray images
- 65,379 unique patients
- 14 thoracic findings (multi-label classification)
- Requires PhysioNet registration
- Real-world hospital data

**Implementation**:
```python
from code.datasets import MIMICCXRDataset, create_mimic_cxr_loader

# Create dataset
dataset = MIMICCXRDataset(
    root_dir='data/mimic_cxr',
    split='train',
    image_size=224,
    normalize=True
)

# Get dataset info
info = MIMICCXRDataset.get_info()
print(f"Total images: {info['total_images']}")
print(f"Findings: {info['findings']}")

# Create dataloader
loader = create_mimic_cxr_loader(
    'data/mimic_cxr',
    split='train',
    batch_size=32,
    num_workers=4
)

for batch in loader:
    images = batch['image']        # (32, 3, 224, 224)
    labels = batch['labels']       # (32, 14) - multi-hot
    patient_id = batch['patient_id']
    break
```

**Key Findings**:
- Common pathologies: Cardiomegaly, Edema, Pneumonia
- Multi-label classification (vs single-label)
- Real clinical scenarios with class imbalance

---

### 1.2 MedMNIST Collection

**File**: `code/datasets/medmnist.py`

**Supports 7 Datasets**:
1. **PathMNIST** - Histopathology (9 classes, 89.996K images)
2. **ChestMNIST** - Chest X-ray (11 classes, 112.009K images)
3. **DermaMNIST** - Skin lesions (7 classes, 10.015K images)
4. **RetinaMNIST** - Fundus photos (5 classes, 9.592K images)
5. **BloodMNIST** - Blood cells (8 classes, 17.092K images)
6. **TissueMNIST** - Histology (8 classes, 236.386K images)
7. **OrganMNIST** - CT organs (11 classes, 58.850K images)

**Implementation**:
```python
from code.datasets import MedMNISTDataset, create_medmnist_loader

# Create dataset
dataset = MedMNISTDataset(
    root_dir='data/medmnist',
    dataset_name='chest',  # or 'path', 'derma', etc.
    split='train',
    image_size=224
)

# Get all dataset info
info = MedMNISTDataset.get_info()
for name, details in info.items():
    print(f"{name}: {details['n_images']} images, {details['n_classes']} classes")

# Create loader
loader = create_medmnist_loader(
    'data/medmnist',
    dataset_name='chest',
    batch_size=32
)
```

**Advantages**:
- Pre-split train/val/test
- Standardized format (28×28 → resize to 224×224)
- Multi-domain evaluation
- Citation-ready datasets

---

### 1.3 CheXpert Dataset

**File**: `code/datasets/chexpert.py`

**Features**:
- 224,316 chest radiographs
- 65,368 patients
- 14 clinical observations
- 5 view types (PA, AP, LATERAL, etc.)
- Uncertainty labels (-1, 0, 1)

**Implementation**:
```python
from code.datasets import CheXpertDataset, create_chexpert_loader

# Create dataset with uncertainty handling
dataset = CheXpertDataset(
    root_dir='data/chexpert',
    split='train',
    uncertainty_handling='positive',  # Treat uncertain as positive
    view='PA'  # Filter to PA view only
)

# Get dataset info
info = CheXpertDataset.get_info()
print(f"Observations: {info['observations']}")
print(f"Views: {info['views']}")

# Create loader
loader = create_chexpert_loader(
    'data/chexpert',
    split='train',
    view='PA',
    batch_size=32
)
```

**Handling Uncertainty**:
```
-1 (uncertain) → Based on uncertainty_handling:
  - 'positive': Treat as 1 (positive)
  - 'negative': Treat as 0 (negative)
  - 'ignore': Treat as 0 (negative)
0 (negative) → 0
1 (positive) → 1
```

---

### 1.4 Federated Data Distribution

**File**: `code/datasets/loader.py`

**Create IID Splits**:
```python
from code.datasets import create_federated_loaders

# IID distribution (each client gets similar data)
loaders = create_federated_loaders(
    dataset,
    num_clients=5,
    batch_size=32,
    split_type='iid'  # Independent and identically distributed
)

# Access per-client data
for client_id, loader in enumerate(loaders):
    print(f"Client {client_id}: {len(loader.dataset)} samples")
```

**Create Non-IID Splits** (more realistic):
```python
# Non-IID distribution (clients have different data distributions)
loaders = create_federated_loaders(
    dataset,
    num_clients=5,
    batch_size=32,
    split_type='non_iid',
    alpha=0.1  # Lower α = more non-IID (more heterogeneous)
)
```

**Dirichlet-based Non-IID**:
- Uses Dirichlet distribution
- α = 0.1: Highly non-IID (each client sees few classes)
- α = 1.0: Somewhat non-IID
- α = ∞: IID (uniform distribution)

---

## 2. Privacy Auditing with MIA Attacks

### 2.1 Membership Inference Attacks

**File**: `code/privacy_audit.py`

**Principle**:
1. Train shadow models on different subsets
2. Record predictions on members vs non-members
3. Train attack classifier on member/non-member losses
4. If DP works → similar loss for both groups
5. If DP fails → members have lower loss (overfitting)

**Implementation**:
```python
from code.privacy_audit import MembershipInferenceAttack, MIAConfig

# Configure MIA
mia_config = MIAConfig(
    n_shadow_models=5,           # Number of shadow models
    shadow_model_epochs=20,      # Training epochs
    use_loss_only=True,          # Use only loss as feature
    device='cuda'
)

# Create MIA
def model_constructor():
    """Function to create new model instances."""
    return FTTransformer(config)

mia = MembershipInferenceAttack(model_constructor, mia_config)

# Train shadow models
mia.train_shadow_models(
    dataset=full_dataset,
    batch_size=32,
    num_workers=4
)

# Evaluate attack
results = mia.evaluate_attack()

print(f"AUC: {results['auc']:.4f}")
print(f"TPR @ FPR=0.5: {results['tpr_at_50_fpr']:.4f}")
print(f"Privacy Level: {results['privacy_level']}")
```

**Interpreting Results**:
- **AUC > 0.6**: Low privacy (Poor DP protection)
- **AUC = 0.55-0.60**: Moderate privacy
- **AUC ≈ 0.5**: High privacy (Strong DP protection)

### 2.2 Privacy Metrics

**Key Metrics**:
1. **AUC** (Area Under Curve): Overall attack success
2. **TPR @ FPR=0.5**: True positive rate when false positive rate is 50%
3. **FPR @ TPR=0.5**: False positive rate when true positive rate is 50%
4. **Member vs Non-member Loss Gap**: Should be small with DP

**ROC Curve Interpretation**:
- Diagonal line (y=x): Random guessing (AUC=0.5) - Good privacy
- Top-left: Perfect attack (AUC=1.0) - Bad privacy

---

## 3. Publication-Ready Visualizations

### 3.1 Visualization Framework

**File**: `code/visualization.py`

**Features**:
- 300 DPI for publication
- Professional matplotlib styling
- Consistent color schemes
- Single/double column layouts

**Set Paper Style**:
```python
from code.visualization import PublicationVisualizer

# Apply publication styling
PublicationVisualizer.set_paper_style()
```

### 3.2 ROC Curves

```python
from code.visualization import PublicationVisualizer

results = {
    'Model A': {'fpr': fpr_a, 'tpr': tpr_a, 'auc': 0.85},
    'Model B': {'fpr': fpr_b, 'tpr': tpr_b, 'auc': 0.82}
}

fig, ax = PublicationVisualizer.plot_roc_curves(
    results,
    title="ROC Curves Comparison",
    save_path="results/roc_curves.pdf"
)
```

**Output**:
- High-quality PDF/PNG (300 DPI)
- Publication-ready legend
- Proper axis labels and spacing

### 3.3 Confusion Matrices

```python
from code.visualization import PublicationVisualizer

fig, ax = PublicationVisualizer.plot_confusion_matrix(
    y_true=test_labels,
    y_pred=predictions,
    class_names=['Normal', 'Abnormal'],
    title="Chest X-ray Classification",
    save_path="results/confusion_matrix.pdf",
    normalize=True
)
```

### 3.4 Privacy-Utility Tradeoff

```python
epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
accuracy = [[0.75, 0.80, 0.82, 0.83, 0.85]]  # per model
attack_success = [[80, 70, 55, 45, 35]]

fig, (ax1, ax2) = PublicationVisualizer.plot_privacy_utility_tradeoff(
    epsilon_values=epsilon_values,
    accuracy_values=accuracy,
    attack_success=attack_success,
    model_names=['FTT + DP'],
    save_path="results/privacy_utility.pdf"
)
```

### 3.5 Communication Rounds

```python
import numpy as np

round_numbers = np.arange(1, 11)
accuracy_values = {
    'FedAvg': np.array([0.60, 0.70, 0.75, 0.78, 0.80, ...]),
    'FedAvg+DP': np.array([0.58, 0.68, 0.72, 0.76, 0.78, ...])
}

fig, ax = PublicationVisualizer.plot_communication_rounds(
    round_numbers=round_numbers,
    accuracy_values=accuracy_values,
    title="Federated Learning: Accuracy vs Communication",
    save_path="results/federated_rounds.pdf"
)
```

### 3.6 Memory Profiling

```python
model_names = ['FT-Transformer', 'ViT-Base', 'ResNet-50']
memory_peak = np.array([1024, 1536, 2048])
memory_mean = np.array([512, 768, 1024])

fig, ax = PublicationVisualizer.plot_memory_profile(
    model_names=model_names,
    memory_peak=memory_peak,
    memory_mean=memory_mean,
    title="Memory Profile Comparison",
    save_path="results/memory_profile.pdf"
)
```

### 3.7 Results Table

```python
results = {
    'ModelA_accuracy': 0.85,
    'ModelA_auc': 0.92,
    'ModelB_accuracy': 0.83,
    'ModelB_auc': 0.90
}

fig, ax = PublicationVisualizer.create_results_summary_table(
    results=results,
    metrics=['accuracy', 'auc'],
    model_names=['Model A', 'Model B'],
    save_path="results/results_table.pdf"
)
```

---

## 4. Experiment Runner

### 4.1 Medical Imaging Experiments

**File**: `scripts/run_medical_imaging_experiments.py`

**Complete Experiment Pipeline**:
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
```

**What It Does**:
1. Loads medical imaging dataset
2. Creates model
3. Trains model with monitoring
4. Evaluates on test set
5. Logs all results
6. Saves checkpoints

---

## 5. Data Download Instructions

### MIMIC-CXR
```bash
# 1. Register at https://physionet.org/
# 2. Get credentialed access
# 3. Download from https://physionet.org/content/mimic-cxr/
# 4. Extract to: data/mimic_cxr/
```

### MedMNIST
```bash
# Auto-download (downloads on first use)
from code.datasets import MedMNISTDataset

dataset = MedMNISTDataset(
    root_dir='data/medmnist',
    dataset_name='chest'
)
# Files auto-download from Zenodo
```

### CheXpert
```bash
# 1. Register at https://stanfordmlgroup.github.io/competitions/chexpert/
# 2. Download CSV metadata
# 3. Download images
# 4. Extract to: data/chexpert/
```

---

## 6. Directory Structure

```
code/
├── datasets/
│   ├── __init__.py              (Package exports)
│   ├── mimic_cxr.py            (MIMIC-CXR loader)
│   ├── medmnist.py             (MedMNIST loaders)
│   ├── chexpert.py             (CheXpert loader)
│   └── loader.py               (Federated splits)
├── privacy_audit.py            (MIA implementation)
└── visualization.py            (Publication visualizations)

scripts/
└── run_medical_imaging_experiments.py  (Experiment runner)

results/
├── {experiment_name}/
│   ├── experiment.log          (Detailed logs)
│   ├── results.json            (Results data)
│   ├── best_model.pth          (Best checkpoint)
│   ├── roc_curves.pdf          (ROC curves - 300 DPI)
│   ├── confusion_matrix.pdf    (Confusion matrix - 300 DPI)
│   └── privacy_utility.pdf     (Privacy-utility tradeoff - 300 DPI)
```

---

## 7. Publication Checklist

### ✅ Experiments
- [ ] Train on MIMIC-CXR
- [ ] Train on MedMNIST (all 7 datasets)
- [ ] Train on CheXpert
- [ ] Privacy auditing (MIA attacks)
- [ ] Federated learning experiments
- [ ] Hyperparameter sensitivity analysis

### ✅ Visualizations
- [ ] ROC curves (all models)
- [ ] Confusion matrices
- [ ] Privacy-utility tradeoff curves
- [ ] Communication rounds plot
- [ ] Memory profiles
- [ ] Results summary tables
- [ ] All figures 300 DPI PNG/PDF

### ✅ Paper Sections
- [ ] Introduction (motivation)
- [ ] Methods (model, DP, federated learning)
- [ ] Datasets (MIMIC-CXR, MedMNIST, CheXpert)
- [ ] Experiments (setup, results)
- [ ] Privacy Analysis (MIA attacks)
- [ ] Results & Discussion
- [ ] Conclusion

---

## 8. Timeline Estimate

- **Week 1-2**: Dataset setup & experiment infrastructure
- **Week 3-4**: Train on multiple datasets
- **Week 5-6**: Privacy auditing & visualization
- **Week 7-8**: Analysis, paper writing, final results

---

## 9. Next Phase

After Medium-term completion, move to:
- **Long-term (9-12 weeks)**: Publication submission & review
- Real deployment & validation
- Open-source release

---

**Status**: Implementation Complete ✅  
**Ready for Execution**: YES  
**Tests Available**: Unit tests created for all modules
