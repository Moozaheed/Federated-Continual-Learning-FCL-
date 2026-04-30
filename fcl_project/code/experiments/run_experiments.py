"""Main Experiment Runner for FCL Publication.

Usage examples:
    # Full experiment grid (all strategies, 3 seeds)
    python -m code.experiments.run_experiments --cl_strategy all --fl_strategy all

    # Quick validation
    python -m code.experiments.run_experiments --cl_strategy finetune ewc \\
        --fl_strategy fedavg --seeds 42 --fl_rounds 5 --local_epochs 2

    # 8-task with MIMIC-CXR
    python -m code.experiments.run_experiments --include_chest --include_mimic \\
        --mimic_data_dir data/mimic --cl_strategy all --fl_strategy all

    # Hyperparameter sensitivity
    python -m code.experiments.run_experiments --mode sensitivity

    # Non-IID analysis
    python -m code.experiments.run_experiments --non_iid_alphas 0.1 0.5 1.0
"""

import argparse
import copy
import csv
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

from code.datasets.medmnist import MedMNISTDataset
from code.experiments.metrics import AccuracyMatrix, evaluate_all_tasks, compute_roc_auc_per_task
from code.experiments.federated import (
    FedAvgServer, FedProxServer, train_client_local, create_dirichlet_splits,
)
from code.experiments.continual import (
    FineTuneStrategy, EWCStrategy, DERStrategy, GenReplayStrategy,
)
from code.utils import get_cosine_warmup_scheduler

logger = logging.getLogger(__name__)

TASK_DATASETS = ['path', 'blood', 'derma', 'retina', 'tissue', 'organ']
TASK_N_CLASSES = [9, 8, 7, 5, 8, 11]


# ---------------------------------------------------------------------------
# Multi-head image classification model
# ---------------------------------------------------------------------------

class MultiHeadImageModel(nn.Module):
    """MobileNetV3-Small backbone with per-task classification heads.

    Shared backbone extracts 576-dim features; separate heads map to
    task-specific label spaces. Supports both single-label (softmax) and
    multi-label (sigmoid) tasks — the head architecture is identical,
    only the loss function differs.
    """

    def __init__(self, task_n_classes: List[int], pretrained: bool = True,
                 task_types: Optional[List[str]] = None):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        self.feature_dim = 576
        self.backbone.classifier = nn.Identity()
        self.task_types = task_types or ['single_label'] * len(task_n_classes)

        self.heads = nn.ModuleDict()
        for i, nc in enumerate(task_n_classes):
            self.heads[f'task_{i}'] = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, nc),
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return feat.view(feat.size(0), -1)

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.heads[f'task_{task_id}'](feat)


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def get_criterion(task_type: str) -> nn.Module:
    """Return correct loss function for task type."""
    if task_type == 'multi_label':
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_task_datasets(
    data_dir: str = 'fcl_project/data/medmnist',
    datasets: Optional[List[str]] = None,
) -> List[Dict]:
    """Load MedMNIST datasets as sequential tasks.

    Returns list of dicts with keys: train, val, test, n_classes, name, task_type.
    """
    if datasets is None:
        datasets = TASK_DATASETS

    tasks = []
    for ds_name in datasets:
        train_ds = MedMNISTDataset(data_dir, dataset_name=ds_name, split='train')
        val_ds = MedMNISTDataset(data_dir, dataset_name=ds_name, split='val')
        test_ds = MedMNISTDataset(data_dir, dataset_name=ds_name, split='test')
        info = MedMNISTDataset.get_info(ds_name)
        task_type = 'multi_label' if info.get('multi_label', False) else 'single_label'
        tasks.append({
            'train': train_ds,
            'val': val_ds,
            'test': test_ds,
            'n_classes': info['n_classes'],
            'name': info['name'],
            'task_type': task_type,
        })
        logger.info(f"Loaded {info['name']}: train={len(train_ds)}, "
                     f"val={len(val_ds)}, test={len(test_ds)}, "
                     f"classes={info['n_classes']}, type={task_type}")
    return tasks


def load_mimic_task(data_dir: str = 'data/mimic') -> Dict:
    """Load MIMIC-CXR as a single multi-label task."""
    from code.datasets.mimic_cxr import MIMICCXRDataset
    train_ds = MIMICCXRDataset(data_dir, split='train')
    val_ds = MIMICCXRDataset(data_dir, split='val')
    test_ds = MIMICCXRDataset(data_dir, split='test')
    info = MIMICCXRDataset.get_info()
    logger.info(f"Loaded MIMIC-CXR: train={len(train_ds)}, "
                f"val={len(val_ds)}, test={len(test_ds)}, "
                f"findings={info['n_findings']}, type=multi_label")
    return {
        'train': train_ds,
        'val': val_ds,
        'test': test_ds,
        'n_classes': info['n_findings'],
        'name': 'MIMIC-CXR',
        'task_type': 'multi_label',
    }


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def make_cl_strategy(name: str, device: str, **kwargs):
    strategies = {
        'finetune': lambda: FineTuneStrategy(),
        'ewc': lambda: EWCStrategy(
            lambda_ewc=kwargs.get('ewc_lambda', 0.5), device=device
        ),
        'der': lambda: DERStrategy(
            buffer_size=kwargs.get('buffer_size', 5000),
            alpha=kwargs.get('der_alpha', 0.3),
            beta=kwargs.get('der_beta', 0.7),
            device=device,
        ),
        'generative_replay': lambda: GenReplayStrategy(
            feature_dim=576, latent_dim=64,
            replay_per_task=kwargs.get('replay_per_task', 500),
            device=device,
        ),
    }
    return strategies[name]()


def make_fl_server(name: str, model: nn.Module, device: str, mu: float = 0.01):
    if name == 'fedavg':
        return FedAvgServer(model, device=device)
    elif name == 'fedprox':
        return FedProxServer(model, mu=mu, device=device)
    raise ValueError(f"Unknown FL strategy: {name}")


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single_experiment(
    seed: int,
    cl_strategy_name: str,
    fl_strategy_name: str,
    alpha: float,
    tasks: List[Dict],
    n_clients: int = 4,
    fl_rounds: int = 20,
    local_epochs: int = 5,
    warmup_epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = 'cpu',
    mu: float = 0.01,
    run_privacy: bool = False,
    **cl_kwargs,
) -> Dict:
    """Run one complete FCL experiment across all tasks."""
    set_seed(seed)
    t0 = time.time()
    n_tasks = len(tasks)
    task_classes = [t['n_classes'] for t in tasks]
    task_types = [t.get('task_type', 'single_label') for t in tasks]

    model = MultiHeadImageModel(task_classes, pretrained=True, task_types=task_types).to(device)
    server = make_fl_server(fl_strategy_name, model, device, mu=mu)
    cl = make_cl_strategy(cl_strategy_name, device, **cl_kwargs)
    acc_matrix = AccuracyMatrix(n_tasks)

    test_loaders = [
        DataLoader(t['test'], batch_size=batch_size, shuffle=False, num_workers=2)
        for t in tasks
    ]

    for task_id in range(n_tasks):
        task_type = task_types[task_id]
        logger.info(f"--- Task {task_id}: {tasks[task_id]['name']} ({task_type}) ---")
        cl.pre_task(model, task_id)

        client_subsets = create_dirichlet_splits(
            tasks[task_id]['train'], n_clients, alpha,
            seed=seed + task_id, task_type=task_type,
        )
        client_loaders = [
            DataLoader(s, batch_size=batch_size, shuffle=True, num_workers=2)
            for s in client_subsets
        ]
        client_weights = [len(s) for s in client_subsets]
        criterion = get_criterion(task_type)

        for fl_round in range(fl_rounds):
            client_models = [copy.deepcopy(server.global_model).to(device) for _ in range(n_clients)]
            server.distribute(client_models)
            global_params = server.get_global_params() if fl_strategy_name == 'fedprox' else None

            client_state_dicts = []
            for cid in range(n_clients):
                cm = client_models[cid]
                opt = torch.optim.Adam(cm.parameters(), lr=lr, weight_decay=1e-4)
                total_ep = local_epochs
                sched = get_cosine_warmup_scheduler(opt, warmup_epochs if task_id == 0 and fl_round == 0 else 0, total_ep)

                sd = train_client_local(
                    cm, client_loaders[cid], opt,
                    criterion, device, total_ep,
                    task_id=task_id,
                    task_type=task_type,
                    global_params=global_params,
                    mu=mu if fl_strategy_name == 'fedprox' else 0.0,
                    scheduler=sched,
                    cl_strategy=cl,
                )
                client_state_dicts.append(sd)

            server.aggregate(client_state_dicts, client_weights)

            if (fl_round + 1) % max(1, fl_rounds // 4) == 0:
                model.load_state_dict(server.global_model.state_dict())
                accs = evaluate_all_tasks(
                    model, test_loaders[:task_id + 1], device,
                    task_types=task_types[:task_id + 1],
                )
                logger.info(f"  Round {fl_round+1}/{fl_rounds} | Scores: {[f'{a:.3f}' for a in accs]}")

        model.load_state_dict(server.global_model.state_dict())
        cl.post_task(model, task_id, client_loaders[0], task_type=task_type)

        accs = evaluate_all_tasks(
            model, test_loaders[:task_id + 1], device,
            task_types=task_types[:task_id + 1],
        )
        for j, a in enumerate(accs):
            acc_matrix.update(task_id, j, a)
        logger.info(f"After task {task_id}: {[f'{a:.3f}' for a in accs]}")

    auc_scores = compute_roc_auc_per_task(
        model, test_loaders, device, task_types=task_types,
    )
    elapsed = time.time() - t0

    result = {
        'seed': seed,
        'cl_strategy': cl_strategy_name,
        'fl_strategy': fl_strategy_name,
        'non_iid_alpha': alpha,
        'n_clients': n_clients,
        'fl_rounds': fl_rounds,
        'local_epochs': local_epochs,
        'task_types': task_types,
        **acc_matrix.to_dict(),
        'roc_auc_per_task': auc_scores,
        'roc_auc_mean': float(np.mean(auc_scores)),
        'training_time_sec': elapsed,
    }
    logger.info(f"Experiment done: BWT={result['bwt']:.4f} FWT={result['fwt']:.4f} "
                f"AvgAcc={result['average_accuracy']:.4f} Time={elapsed:.0f}s")
    return result


# ---------------------------------------------------------------------------
# Experiment runner (grid)
# ---------------------------------------------------------------------------

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = []
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / 'experiment.log'),
            ],
        )

    def run_all(self):
        tasks = load_task_datasets(self.args.data_dir, self.args.datasets)

        if getattr(self.args, 'include_chest', False) and 'chest' not in self.args.datasets:
            chest_tasks = load_task_datasets(self.args.data_dir, ['chest'])
            tasks.extend(chest_tasks)

        if getattr(self.args, 'include_mimic', False):
            mimic_dir = getattr(self.args, 'mimic_data_dir', 'data/mimic')
            tasks.append(load_mimic_task(mimic_dir))

        total = (len(self.args.seeds) * len(self.args.cl_strategy)
                 * len(self.args.fl_strategy) * len(self.args.non_iid_alphas))
        logger.info(f"Running {total} experiments on {self.device} "
                     f"with {len(tasks)} tasks: {[t['name'] for t in tasks]}")

        idx = 0
        for seed in self.args.seeds:
            for cl in self.args.cl_strategy:
                for fl in self.args.fl_strategy:
                    for alpha in self.args.non_iid_alphas:
                        idx += 1
                        logger.info(f"\n{'='*60}")
                        logger.info(f"Experiment {idx}/{total}: seed={seed} CL={cl} FL={fl} alpha={alpha}")
                        result = run_single_experiment(
                            seed=seed, cl_strategy_name=cl, fl_strategy_name=fl,
                            alpha=alpha, tasks=tasks,
                            n_clients=self.args.n_clients,
                            fl_rounds=self.args.fl_rounds,
                            local_epochs=self.args.local_epochs,
                            warmup_epochs=self.args.warmup_epochs,
                            batch_size=self.args.batch_size,
                            device=self.device,
                            run_privacy=self.args.run_privacy,
                        )
                        self.results.append(result)
        self.aggregate_and_save()

    def aggregate_and_save(self):
        """Compute mean +/- std across seeds and save results."""
        grouped = defaultdict(list)
        for r in self.results:
            key = (r['cl_strategy'], r['fl_strategy'], r['non_iid_alpha'])
            grouped[key].append(r)

        rows = []
        for (cl, fl, alpha), runs in sorted(grouped.items()):
            bwts = [r['bwt'] for r in runs]
            fwts = [r['fwt'] for r in runs]
            avg_accs = [r['average_accuracy'] for r in runs]
            forgets = [r['forgetting'] for r in runs]
            aucs = [r['roc_auc_mean'] for r in runs]
            times = [r['training_time_sec'] for r in runs]

            rows.append({
                'cl_strategy': cl,
                'fl_strategy': fl,
                'non_iid_alpha': alpha,
                'n_seeds': len(runs),
                'avg_accuracy_mean': f"{np.mean(avg_accs):.4f}",
                'avg_accuracy_std': f"{np.std(avg_accs):.4f}",
                'bwt_mean': f"{np.mean(bwts):.4f}",
                'bwt_std': f"{np.std(bwts):.4f}",
                'fwt_mean': f"{np.mean(fwts):.4f}",
                'fwt_std': f"{np.std(fwts):.4f}",
                'forgetting_mean': f"{np.mean(forgets):.4f}",
                'forgetting_std': f"{np.std(forgets):.4f}",
                'roc_auc_mean': f"{np.mean(aucs):.4f}",
                'roc_auc_std': f"{np.std(aucs):.4f}",
                'time_mean_sec': f"{np.mean(times):.0f}",
            })

        csv_path = self.output_dir / 'results_summary.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Summary saved to {csv_path}")

        json_path = self.output_dir / 'results_detailed.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {json_path}")

        self._generate_figures()

    def _generate_figures(self):
        """Generate publication-quality figures from results."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping figures")
            return

        fig_dir = self.output_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)

        grouped = defaultdict(list)
        for r in self.results:
            key = (r['cl_strategy'], r['fl_strategy'], r['non_iid_alpha'])
            grouped[key].append(r)

        # Figure 1: Accuracy matrix heatmaps (last seed of each strategy combo)
        for (cl, fl, alpha), runs in grouped.items():
            matrix = np.array(runs[-1]['matrix'])
            fig, ax = plt.subplots(figsize=(max(5, matrix.shape[1] * 0.8), max(4, matrix.shape[0] * 0.7)))
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                        xticklabels=[f'Task {i}' for i in range(matrix.shape[1])],
                        yticklabels=[f'After {i}' for i in range(matrix.shape[0])],
                        ax=ax)
            ax.set_title(f'{cl.upper()} + {fl.upper()} (alpha={alpha})')
            ax.set_xlabel('Evaluated on Task')
            ax.set_ylabel('Trained through Task')
            plt.tight_layout()
            plt.savefig(fig_dir / f'acc_matrix_{cl}_{fl}_a{alpha}.png', dpi=300)
            plt.close()

        # Figure 2: BWT/FWT comparison bar chart
        cl_names = sorted(set(r['cl_strategy'] for r in self.results))
        bwt_means = []
        fwt_means = []
        for cl in cl_names:
            bwts = [r['bwt'] for r in self.results if r['cl_strategy'] == cl]
            fwts = [r['fwt'] for r in self.results if r['cl_strategy'] == cl]
            bwt_means.append(np.mean(bwts))
            fwt_means.append(np.mean(fwts))

        x = np.arange(len(cl_names))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - 0.2, bwt_means, 0.35, label='BWT', color='steelblue')
        ax.bar(x + 0.2, fwt_means, 0.35, label='FWT', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels([n.upper() for n in cl_names])
        ax.set_ylabel('Transfer Score')
        ax.set_title('Backward/Forward Transfer by CL Strategy')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(fig_dir / 'bwt_fwt_comparison.png', dpi=300)
        plt.close()

        # Figure 3: Average accuracy comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        for cl in cl_names:
            accs = [r['average_accuracy'] for r in self.results if r['cl_strategy'] == cl]
            ax.bar(cl.upper(), np.mean(accs), yerr=np.std(accs), capsize=5)
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Average Accuracy Across All Tasks')
        plt.tight_layout()
        plt.savefig(fig_dir / 'average_accuracy.png', dpi=300)
        plt.close()

        logger.info(f"Figures saved to {fig_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='FCL Experiment Runner for Q1 Journal Publication'
    )
    parser.add_argument('--data_dir', type=str, default='fcl_project/data/medmnist')
    parser.add_argument('--datasets', nargs='+',
                        default=['path', 'blood', 'derma', 'retina', 'tissue', 'organ'])
    parser.add_argument('--include_chest', action='store_true',
                        help='Add ChestMNIST (multi-label) as additional task')
    parser.add_argument('--include_mimic', action='store_true',
                        help='Add MIMIC-CXR (multi-label) as final task')
    parser.add_argument('--mimic_data_dir', type=str, default='data/mimic',
                        help='Path to MIMIC-CXR data directory')
    parser.add_argument('--cl_strategy', nargs='+', default=['all'])
    parser.add_argument('--fl_strategy', nargs='+', default=['all'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--n_clients', type=int, default=4)
    parser.add_argument('--non_iid_alphas', nargs='+', type=float, default=[0.5])
    parser.add_argument('--fl_rounds', type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'sensitivity'])
    parser.add_argument('--run_privacy', action='store_true')

    args = parser.parse_args()

    if 'all' in args.cl_strategy:
        args.cl_strategy = ['finetune', 'ewc', 'der', 'generative_replay']
    if 'all' in args.fl_strategy:
        args.fl_strategy = ['fedavg', 'fedprox']

    if args.mode == 'sensitivity':
        from code.experiments.analysis import HyperparameterAnalysis
        analysis = HyperparameterAnalysis(args)
        analysis.run_all()
    else:
        runner = ExperimentRunner(args)
        runner.run_all()


if __name__ == '__main__':
    main()
