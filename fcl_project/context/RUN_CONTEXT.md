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
│   │   ├── mimic_cxr.py           # MIMIC-CXR loader
│   │   ├── chexpert.py            # CheXpert loader
│   │   └── loader.py              # Federated split utilities
│   ├── multimodal/
│   │   ├── image_extractor.py     # MobileNetV3 feature extractor
│   │   └── fusion.py              # Multimodal fusion layer
│   └── experiments/               # <-- NEW: Main experiment runner
│       ├── run_experiments.py     # CLI entry point
│       ├── metrics.py             # AccuracyMatrix, BWT, FWT, forgetting
│       ├── federated.py           # FedAvg + FedProx servers
│       ├── continual.py           # EWC, DER++, GenReplay, FineTune
│       └── analysis.py            # Hyperparameter sensitivity sweeps
├── data/
│   └── medmnist/                  # 7 NPZ files (already downloaded)
│       ├── pathmnist.npz          # PathMNIST (9 classes, 197 MB)
│       ├── bloodmnist.npz         # BloodMNIST (8 classes, 35 MB)
│       ├── dermamnist.npz         # DermaMNIST (7 classes, 19 MB)
│       ├── chestmnist.npz
│       ├── tissuemnist.npz
│       ├── organmnist.npz
│       └── retinamnist.npz
├── tests/unit/                    # Unit tests (pytest)
├── requirements.txt
└── RUN_CONTEXT.md                 # This file
```

## 3. GPU and Memory Requirements

| Experiment | VRAM | RAM | Time (est.) |
|------------|------|-----|-------------|
| Single run (1 seed, 1 strategy) | ~4 GB | ~8 GB | ~30 min |
| Full grid (3 seeds x 4 CL x 2 FL) | ~4 GB | ~8 GB | ~12 hours |
| Sensitivity sweeps | ~4 GB | ~8 GB | ~24 hours |
| DER++ feature buffer (5000 samples) | +0 GPU | +12 MB | negligible |

The DER++ buffer stores 576-dim features (not raw images), keeping memory low.

## 4. Running Experiments

All commands assume you are in the `fcl_project/` directory.

### 4.1 Quick Validation (run first to verify setup)

```bash
python -m code.experiments.run_experiments \
    --data_dir data/medmnist \
    --cl_strategy finetune ewc \
    --fl_strategy fedavg \
    --seeds 42 \
    --fl_rounds 3 \
    --local_epochs 2 \
    --output_dir results/quick_test
```

Expected: completes in ~10 minutes, produces `results/quick_test/results_summary.csv`.

### 4.2 Main Experiment Grid (primary results for paper)

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
    --output_dir results/main_experiment
```

This runs: 3 seeds x 4 CL strategies x 2 FL strategies = **24 experiments**.

Tasks: PathMNIST (9 classes) -> BloodMNIST (8 classes) -> DermaMNIST (7 classes).
Model: MobileNetV3-Small backbone (pretrained) with per-task classification heads.
FL: 4 clients with Dirichlet non-IID (alpha=0.5), 20 rounds, 5 local epochs each.
CL: FineTune, EWC (lambda=0.5), DER++ (alpha=0.3, beta=0.7, 5000 buffer), GenReplay (VAE).

### 4.3 Non-IID Analysis (heterogeneity effect)

```bash
python -m code.experiments.run_experiments \
    --data_dir data/medmnist \
    --cl_strategy ewc der \
    --fl_strategy fedavg \
    --seeds 42 123 456 \
    --non_iid_alphas 0.1 0.5 1.0 \
    --fl_rounds 20 \
    --local_epochs 5 \
    --output_dir results/non_iid_analysis
```

### 4.4 Hyperparameter Sensitivity

```bash
python -m code.experiments.run_experiments \
    --data_dir data/medmnist \
    --mode sensitivity \
    --output_dir results/sensitivity
```

Sweeps: EWC lambda (5 values), non-IID alpha (3 values), FL rounds (4 values).

### 4.5 Run Unit Tests

```bash
python -m pytest tests/ -v --tb=short
```

All tests should pass. If any fail, fix before running experiments.

## 5. Expected Outputs

After the main experiment:

```
results/main_experiment/
├── results_summary.csv           # <-- MAIN TABLE for paper
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

## 6. What Metrics to Report in the Paper

From `results_summary.csv`, build Table 1:

| CL Strategy | FL Strategy | Avg Acc | BWT | FWT | Forgetting | ROC-AUC |
|-------------|-------------|---------|-----|-----|------------|---------|
| FineTune | FedAvg | X +/- Y | ... | ... | ... | ... |
| EWC | FedAvg | ... | ... | ... | ... | ... |
| DER++ | FedAvg | ... | ... | ... | ... | ... |
| GenReplay | FedAvg | ... | ... | ... | ... | ... |
| FineTune | FedProx | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

Key findings to highlight:
- DER++ should show best BWT (least forgetting)
- EWC provides good balance of accuracy and stability
- FedProx should improve over FedAvg on non-IID data
- FineTune is the forgetting baseline (worst BWT)

## 7. Troubleshooting

**CUDA OOM**: Reduce `--batch_size 16` and restart.

**Slow training**: Use `--fl_rounds 10 --local_epochs 3` for faster iterations.

**MedMNIST not found**: Verify `data/medmnist/pathmnist.npz` exists. If not:
```bash
python scripts/download_datasets.py
```

**Import errors**: Ensure you run from the `fcl_project/` directory, not from within `code/`.

**MobileNetV3 download fails** (no internet): Pre-download on a connected machine:
```python
import torchvision.models as models
m = models.mobilenet_v3_small(pretrained=True)
torch.save(m.state_dict(), 'mobilenet_v3_small_pretrained.pth')
```
Then pass `--no_pretrained` or modify the code to load from local path.

## 8. Order of Execution

1. Run tests: `python -m pytest tests/ -v`
2. Quick validation (section 4.1)
3. Main experiment grid (section 4.2) -- this produces the primary results
4. Non-IID analysis (section 4.3) -- for the heterogeneity discussion
5. Sensitivity sweeps (section 4.4) -- for the ablation section

## 9. Research Contribution Statement

"We propose the first multimodal federated continual learning framework for medical imaging that combines feature-level DER++ replay with differential privacy, evaluated on clinically realistic non-IID distributions across three medical imaging domains (PathMNIST, BloodMNIST, DermaMNIST) with four federated clients."

Target journals: IEEE JBHI, IEEE TNNLS, Nature Digital Medicine.
