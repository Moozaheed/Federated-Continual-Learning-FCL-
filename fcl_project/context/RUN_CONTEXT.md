# RUN_CONTEXT.md — Remote Execution Guide for FCL Experiments

This file is the complete context for an AI agent running experiments on a remote machine. Follow it step by step.

## 1. Environment Setup

```bash
conda create -n fcl python=3.10 -y
conda activate fcl
pip install -r requirements.txt
```

Verify GPU:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## 2. Project Structure

```
fcl_project/
├── code/
│   ├── config.py                  # All configuration classes
│   ├── model.py                   # FT-Transformer (tabular model)
│   ├── utils.py                   # Training utilities, metrics, LR scheduler
│   ├── der.py                     # DER++ replay buffer
│   ├── privacy_audit.py           # Membership Inference Attack evaluation
│   ├── visualization.py           # Publication-grade 300 DPI figures
│   ├── datasets/
│   │   ├── medmnist.py            # MedMNIST loader (7 datasets)
│   │   ├── mimic_cxr.py           # MIMIC-CXR loader (multi-label)
│   │   ├── chexpert.py            # CheXpert loader
│   │   └── loader.py              # Federated split utilities
│   ├── multimodal/
│   │   ├── image_extractor.py     # MobileNetV3 feature extractor
│   │   └── fusion.py              # Multimodal fusion layer
│   └── experiments/
│       ├── run_experiments.py     # CLI entry point
│       ├── metrics.py             # AccuracyMatrix, BWT, FWT, forgetting
│       ├── federated.py           # FedAvg + FedProx servers
│       ├── continual.py           # EWC, DER++, GenReplay, FineTune
│       └── analysis.py            # Hyperparameter sensitivity sweeps
├── data/
│   ├── medmnist/                  # 7 NPZ files (~487 MB)
│   │   ├── pathmnist.npz          # PathMNIST (9 classes, single-label)
│   │   ├── bloodmnist.npz         # BloodMNIST (8 classes, single-label)
│   │   ├── dermamnist.npz         # DermaMNIST (7 classes, single-label)
│   │   ├── retinamnist.npz        # RetinaMNIST (5 classes, single-label)
│   │   ├── tissuemnist.npz        # TissueMNIST (8 classes, single-label)
│   │   ├── organmnist.npz         # OrganMNIST (11 classes, single-label)
│   │   └── chestmnist.npz         # ChestMNIST (14 findings, multi-label)
│   └── mimic/                     # MIMIC-CXR (~20 GB)
│       ├── train_metadata.json    # Training split metadata
│       ├── val_metadata.json      # Validation split metadata
│       ├── test_metadata.json     # Test split metadata
│       ├── labels.csv             # 14 clinical findings
│       └── files/                 # JPEG images (p10-p19 patient dirs)
├── tests/unit/                    # 166 unit tests (pytest)
├── requirements.txt
└── context/
    └── RUN_CONTEXT.md             # This file
```

## 3. Task Design

### Single-Label Tasks (6) — MedMNIST
Learned sequentially as a continual learning stream:

| Order | Dataset | Classes | Metric | Loss |
|-------|---------|---------|--------|------|
| 1 | PathMNIST | 9 | Accuracy | CrossEntropy |
| 2 | BloodMNIST | 8 | Accuracy | CrossEntropy |
| 3 | DermaMNIST | 7 | Accuracy | CrossEntropy |
| 4 | RetinaMNIST | 5 | Accuracy | CrossEntropy |
| 5 | TissueMNIST | 8 | Accuracy | CrossEntropy |
| 6 | OrganMNIST | 11 | Accuracy | CrossEntropy |

### Multi-Label Tasks (2) — Appended after single-label stream
| Order | Dataset | Findings | Metric | Loss |
|-------|---------|----------|--------|------|
| 7 | ChestMNIST | 14 | Mean AUC-ROC | BCEWithLogits |
| 8 | MIMIC-CXR | 14 | Mean AUC-ROC | BCEWithLogits |

Multi-label tasks use per-finding AUC-ROC (macro-averaged) instead of accuracy. Labels are binary multi-hot vectors of shape (N, 14).

## 4. GPU and Memory Requirements

| Experiment | VRAM | RAM | Time (est.) |
|------------|------|-----|-------------|
| Quick validation (2 tasks, 1 seed) | ~4 GB | ~8 GB | ~10 min |
| 6-task single-label (3 seeds, full grid) | ~4 GB | ~8 GB | ~6 hours |
| 8-task with ChestMNIST (3 seeds, full grid) | ~4 GB | ~10 GB | ~12 hours |
| 8-task with MIMIC-CXR (3 seeds, full grid) | ~6 GB | ~16 GB | ~24 hours |
| Sensitivity sweeps | ~4 GB | ~8 GB | ~24 hours |
| DER++ feature buffer (5000 samples) | +0 GPU | +12 MB | negligible |

MIMIC-CXR has 259K training images (20 GB on disk). Loading uses lazy JPEG reading, so RAM usage stays manageable, but disk I/O is the bottleneck.

## 5. Running Experiments

All commands assume you are in the `fcl_project/` directory.

### 5.1 Run Unit Tests First

```bash
python -m pytest tests/ -v --tb=short
```

All 166 tests should pass. If any fail, fix before running experiments.

### 5.2 Quick Validation (~10 min)

```bash
python -m code.experiments.run_experiments \
    --data_dir data/medmnist \
    --datasets path blood \
    --cl_strategy finetune ewc \
    --fl_strategy fedavg \
    --seeds 42 \
    --fl_rounds 3 \
    --local_epochs 2 \
    --output_dir results/quick_test
```

Expected: completes in ~10 minutes, produces `results/quick_test/results_summary.csv`.

### 5.3 6-Task Single-Label MedMNIST (~6 hours)

The core single-label experiment — 6 datasets, full CL/FL grid:

```bash
python -m code.experiments.run_experiments \
    --data_dir data/medmnist \
    --cl_strategy all \
    --fl_strategy all \
    --seeds 42 123 456 \
    --n_clients 4 \
    --non_iid_alphas 0.5 \
    --fl_rounds 20 \
    --local_epochs 5 \
    --warmup_epochs 5 \
    --output_dir results/6task_experiment
```

Runs: 3 seeds x 4 CL strategies x 2 FL strategies = **24 experiments**.
Tasks: PathMNIST(9) → BloodMNIST(8) → DermaMNIST(7) → RetinaMNIST(5) → TissueMNIST(8) → OrganMNIST(11).

### 5.4 8-Task Full Grid with ChestMNIST + MIMIC-CXR (~24 hours)

The flagship experiment — all 8 tasks including multi-label:

```bash
python -m code.experiments.run_experiments \
    --data_dir data/medmnist \
    --include_chest \
    --include_mimic \
    --mimic_data_dir data/mimic \
    --cl_strategy all \
    --fl_strategy all \
    --seeds 42 123 456 \
    --n_clients 4 \
    --non_iid_alphas 0.5 \
    --fl_rounds 20 \
    --local_epochs 5 \
    --warmup_epochs 5 \
    --output_dir results/8task_experiment
```

Tasks: 6 single-label MedMNIST → ChestMNIST(14, multi-label) → MIMIC-CXR(14, multi-label).

The pipeline automatically selects the correct loss function (CrossEntropy vs BCEWithLogits) and evaluation metric (accuracy vs mean AUC-ROC) per task based on its `task_type`.

### 5.5 7-Task with ChestMNIST Only (no MIMIC-CXR)

If MIMIC-CXR data is not available or you want a faster multi-label run:

```bash
python -m code.experiments.run_experiments \
    --data_dir data/medmnist \
    --include_chest \
    --cl_strategy all \
    --fl_strategy all \
    --seeds 42 123 456 \
    --n_clients 4 \
    --non_iid_alphas 0.5 \
    --fl_rounds 20 \
    --local_epochs 5 \
    --output_dir results/7task_experiment
```

### 5.6 Non-IID Analysis (heterogeneity effect)

```bash
python -m code.experiments.run_experiments \
    --data_dir data/medmnist \
    --include_chest \
    --include_mimic \
    --mimic_data_dir data/mimic \
    --cl_strategy ewc der \
    --fl_strategy fedavg \
    --seeds 42 123 456 \
    --non_iid_alphas 0.1 0.5 1.0 \
    --fl_rounds 20 \
    --local_epochs 5 \
    --output_dir results/non_iid_analysis
```

Three alpha values: 0.1 (highly non-IID) → 0.5 (moderate) → 1.0 (near-IID).

### 5.7 Hyperparameter Sensitivity

```bash
python -m code.experiments.run_experiments \
    --data_dir data/medmnist \
    --mode sensitivity \
    --output_dir results/sensitivity
```

Sweeps: EWC lambda (5 values), non-IID alpha (3 values), FL rounds (4 values).

## 6. Expected Outputs

After the main experiment:

```
results/<experiment_name>/
├── results_summary.csv           # MAIN TABLE for paper
│   Columns: cl_strategy, fl_strategy, non_iid_alpha,
│            avg_accuracy_mean, avg_accuracy_std,
│            bwt_mean, bwt_std, fwt_mean, fwt_std,
│            forgetting_mean, forgetting_std,
│            roc_auc_mean, roc_auc_std, time_mean_sec
├── results_detailed.json         # Full per-seed results with accuracy matrices
├── experiment.log                # Training logs
└── figures/
    ├── acc_matrix_*.png          # Accuracy matrix heatmaps (one per strategy)
    ├── bwt_fwt_comparison.png    # BWT/FWT bar chart
    └── average_accuracy.png      # Average accuracy comparison
```

## 7. What Metrics to Report in the Paper

From `results_summary.csv`, build Table 1:

| CL Strategy | FL Strategy | Avg Acc | BWT | FWT | Forgetting | ROC-AUC |
|-------------|-------------|---------|-----|-----|------------|---------|
| FineTune | FedAvg | X ± Y | ... | ... | ... | ... |
| EWC | FedAvg | ... | ... | ... | ... | ... |
| DER++ | FedAvg | ... | ... | ... | ... | ... |
| GenReplay | FedAvg | ... | ... | ... | ... | ... |
| FineTune | FedProx | ... | ... | ... | ... | ... |
| EWC | FedProx | ... | ... | ... | ... | ... |
| DER++ | FedProx | ... | ... | ... | ... | ... |
| GenReplay | FedProx | ... | ... | ... | ... | ... |

For the 8-task experiment, the accuracy matrix is 8x8. Single-label tasks report accuracy; multi-label tasks (ChestMNIST, MIMIC-CXR) report mean AUC-ROC. BWT and FWT are computed over the full 8x8 matrix.

Key findings to highlight:
- DER++ should show best BWT (least forgetting) due to feature-level replay
- EWC provides good balance of accuracy and stability
- FedProx should improve over FedAvg on non-IID data (Dirichlet alpha=0.5)
- FineTune is the catastrophic forgetting baseline (worst BWT)
- Multi-label tasks (ChestMNIST, MIMIC-CXR) test generalization to clinical multi-finding detection
- MIMIC-CXR (259K images) demonstrates scalability to real-world clinical datasets

## 8. Multi-Label Implementation Details

The pipeline automatically handles mixed single-label / multi-label task sequences:

- **Loss**: `CrossEntropyLoss` for single-label, `BCEWithLogitsLoss` for multi-label
- **Labels**: Single-label uses `batch['label']` (long tensor), multi-label uses `batch['labels']` (float multi-hot tensor)
- **Evaluation**: Accuracy for single-label, per-finding AUC-ROC (macro-averaged) for multi-label
- **DER++ buffer**: Stores all labels as single-label (argmax for multi-label) since the primary DER++ signal is logit matching
- **Dirichlet splits**: Multi-label tasks use argmax of multi-hot as pseudo-class for Dirichlet allocation
- **EWC Fisher**: Computed with BCEWithLogitsLoss for multi-label tasks
- **GenReplay VAE**: Trains on 576-dim features regardless of task type

## 9. Troubleshooting

**CUDA OOM**: Reduce `--batch_size 16` and restart.

**Slow training**: Use `--fl_rounds 10 --local_epochs 3` for faster iterations.

**MedMNIST not found**: Verify `data/medmnist/pathmnist.npz` exists. If not:
```bash
python scripts/download_datasets.py
```

**MIMIC-CXR not found**: Ensure `data/mimic/train_metadata.json` and `data/mimic/files/` exist. MIMIC-CXR must be obtained from [PhysioNet](https://physionet.org/content/mimic-cxr/) and placed in `data/mimic/`.

**Import errors**: Ensure you run from the `fcl_project/` directory, not from within `code/`.

**MobileNetV3 download fails** (no internet): Pre-download on a connected machine:
```python
import torchvision.models as models
m = models.mobilenet_v3_small(pretrained=True)
torch.save(m.state_dict(), 'mobilenet_v3_small_pretrained.pth')
```
Then pass `--no_pretrained` or modify the code to load from local path.

**Multi-label AUC = 0.0**: This happens when a finding has only one class in the test batch (all 0s or all 1s). The pipeline skips such findings and averages over the rest. With sufficient data this resolves itself.

## 10. Order of Execution (Recommended)

1. Run tests: `python -m pytest tests/ -v` (166 should pass)
2. Quick validation (section 5.2) — verify setup works
3. 6-task single-label (section 5.3) — core results, ~6 hours
4. 8-task full grid (section 5.4) — flagship results with MIMIC-CXR, ~24 hours
5. Non-IID analysis (section 5.6) — heterogeneity discussion
6. Sensitivity sweeps (section 5.7) — ablation section

If time-constrained, prioritize steps 1-4. The 8-task experiment is the strongest result for Q1 journal submission.

## 11. Research Contribution Statement

"We propose the first multimodal federated continual learning framework for medical imaging that combines feature-level DER++ replay with differential privacy, evaluated on clinically realistic non-IID distributions across eight tasks: six single-label MedMNIST datasets (PathMNIST, BloodMNIST, DermaMNIST, RetinaMNIST, TissueMNIST, OrganMNIST), multi-label ChestMNIST (14 thoracic findings), and MIMIC-CXR (14 clinical findings, 259K images) with four federated clients."

Target journals: IEEE JBHI, IEEE TNNLS, Nature Digital Medicine.
