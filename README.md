# Federated Continual Learning for Medical Imaging

A multimodal federated continual learning framework combining feature-level DER++ replay with differential privacy, evaluated on clinically realistic non-IID distributions across eight medical imaging tasks spanning seven domains.

## Research Contribution

> We propose the first multimodal federated continual learning framework for medical imaging that combines feature-level DER++ replay with differential privacy, evaluated on clinically realistic non-IID distributions across eight tasks: six single-label MedMNIST datasets (PathMNIST, BloodMNIST, DermaMNIST, RetinaMNIST, TissueMNIST, OrganMNIST), multi-label ChestMNIST (14 thoracic findings), and MIMIC-CXR (14 clinical findings, 259K images) with four federated clients.

**Target venues:** IEEE JBHI, IEEE TNNLS, Nature Digital Medicine

## Project Structure

```
fcl_project/
├── requirements.txt
├── context/
│   └── RUN_CONTEXT.md          # Full experiment execution guide
├── code/
│   ├── config.py                # All configuration classes
│   ├── model.py                 # FT-Transformer (tabular model)
│   ├── utils.py                 # Training utilities, metrics, LR scheduler
│   ├── der.py                   # DER++ replay buffer
│   ├── privacy_audit.py         # Membership Inference Attack evaluation
│   ├── visualization.py         # Publication-grade 300 DPI figures
│   ├── datasets/
│   │   ├── medmnist.py          # MedMNIST loader (7 datasets)
│   │   ├── mimic_cxr.py         # MIMIC-CXR loader (multi-label)
│   │   ├── chexpert.py          # CheXpert loader
│   │   └── loader.py            # Federated split utilities
│   ├── multimodal/
│   │   ├── image_extractor.py   # MobileNetV3 feature extractor
│   │   └── fusion.py            # Multimodal fusion layer
│   └── experiments/
│       ├── run_experiments.py   # CLI entry point
│       ├── metrics.py           # AccuracyMatrix, BWT, FWT, forgetting
│       ├── federated.py         # FedAvg + FedProx servers
│       ├── continual.py         # EWC, DER++, GenReplay, FineTune
│       └── analysis.py          # Hyperparameter sensitivity sweeps
├── scripts/
│   └── download_datasets.py     # Download MedMNIST datasets
└── tests/unit/                  # 166 unit tests
```

## Setup

```bash
conda create -n fcl python=3.10 -y
conda activate fcl
pip install -r fcl_project/requirements.txt
```

## Download Data

```bash
cd fcl_project
python scripts/download_datasets.py
```

Downloads 7 MedMNIST datasets (~500 MB) into `data/medmnist/`.

MIMIC-CXR data (20 GB) must be obtained separately from [PhysioNet](https://physionet.org/content/mimic-cxr/) and placed in `data/mimic/`.

## Run Tests

```bash
cd fcl_project
python -m pytest tests/ -v
```

All 166 tests should pass.

## Run Experiments

### Quick Validation (~10 min)

```bash
cd fcl_project
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

### 6-Task Single-Label MedMNIST (~6 hours)

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

### 8-Task Full Grid with MIMIC-CXR (~24 hours)

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
    --output_dir results/8task_experiment
```

See [`fcl_project/context/RUN_CONTEXT.md`](fcl_project/context/RUN_CONTEXT.md) for the complete execution guide including sensitivity sweeps, non-IID analysis, expected outputs, and troubleshooting.

## Key Components

| Component | Strategy | Reference |
|-----------|----------|-----------|
| Backbone | MobileNetV3-Small (576-dim features) | Howard et al., 2019 |
| Continual Learning | EWC, DER++ (feature-level), Generative Replay | Kirkpatrick 2017, Buzzega 2021 |
| Federated Learning | FedAvg, FedProx | McMahan 2017, Li 2020 |
| Non-IID Splits | Dirichlet distribution (alpha=0.1, 0.5, 1.0) | Hsu et al., 2019 |
| Privacy Audit | Membership Inference Attack | Shokri et al., 2017 |
| Metrics | Accuracy Matrix, BWT, FWT, Forgetting | Lopez-Paz & Ranzato, 2017 |

## Experiment Design

**Single-label tasks (6):** PathMNIST (9 classes) &rarr; BloodMNIST (8) &rarr; DermaMNIST (7) &rarr; RetinaMNIST (5) &rarr; TissueMNIST (8) &rarr; OrganMNIST (11)

**Multi-label tasks (2):** ChestMNIST (14 findings) &rarr; MIMIC-CXR (14 findings, 259K images)

**Federation:** 4 clients with Dirichlet non-IID data distribution

**Full grid:** 3 seeds x 4 CL strategies x 2 FL strategies = 24 experiments per configuration

## References

1. Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS.
2. McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.
3. Buzzega, P. et al. (2021). Dark Experience for General Continual Learning: a Strong, Simple Baseline. NeurIPS.
4. Li, T. et al. (2020). Federated Optimization in Heterogeneous Networks. MLSys.
5. Lopez-Paz, D. & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning. NeurIPS.
6. Shokri, R. et al. (2017). Membership Inference Attacks Against Machine Learning Models. IEEE S&P.

## License

This project is for academic research purposes.
