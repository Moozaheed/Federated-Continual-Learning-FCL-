"""Hyperparameter sensitivity analysis.

Sweeps EWC lambda, non-IID alpha, and FL rounds to produce
sensitivity plots for the publication.
"""

import copy
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class HyperparameterAnalysis:
    """Run parameter sweeps and generate sensitivity plots."""

    def __init__(self, base_args):
        self.base_args = base_args
        self.output_dir = Path(base_args.output_dir) / 'sensitivity'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self):
        results = {}
        results['ewc_lambda'] = self.sweep_ewc_lambda()
        results['non_iid_alpha'] = self.sweep_non_iid_alpha()
        results['fl_rounds'] = self.sweep_fl_rounds()

        json_path = self.output_dir / 'sensitivity_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Sensitivity results saved to {json_path}")

        self.generate_plots(results)

    def _run_sweep(self, param_name: str, param_values: list, **override_kwargs) -> Dict:
        from code.experiments.run_experiments import (
            run_single_experiment, load_task_datasets, set_seed,
        )
        import torch

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tasks = load_task_datasets(self.base_args.data_dir)
        seed = self.base_args.seeds[0]

        sweep_results = {}
        for val in param_values:
            logger.info(f"Sweep {param_name}={val}")
            kwargs = {
                'seed': seed,
                'cl_strategy_name': 'ewc',
                'fl_strategy_name': 'fedavg',
                'alpha': 0.5,
                'tasks': tasks,
                'n_clients': self.base_args.n_clients,
                'fl_rounds': self.base_args.fl_rounds,
                'local_epochs': self.base_args.local_epochs,
                'warmup_epochs': self.base_args.warmup_epochs,
                'batch_size': self.base_args.batch_size,
                'device': device,
            }
            kwargs.update(override_kwargs)

            if param_name == 'ewc_lambda':
                kwargs['ewc_lambda'] = val
            elif param_name == 'non_iid_alpha':
                kwargs['alpha'] = val
            elif param_name == 'fl_rounds':
                kwargs['fl_rounds'] = val

            result = run_single_experiment(**kwargs)
            sweep_results[str(val)] = {
                'avg_accuracy': result['average_accuracy'],
                'bwt': result['bwt'],
                'forgetting': result['forgetting'],
            }
        return sweep_results

    def sweep_ewc_lambda(self, values=None):
        if values is None:
            values = [0.01, 0.1, 0.5, 1.0, 5.0]
        return self._run_sweep('ewc_lambda', values)

    def sweep_non_iid_alpha(self, values=None):
        if values is None:
            values = [0.1, 0.5, 1.0]
        return self._run_sweep('non_iid_alpha', values)

    def sweep_fl_rounds(self, values=None):
        if values is None:
            values = [5, 10, 20, 50]
        return self._run_sweep('fl_rounds', values)

    def generate_plots(self, all_results: Dict):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
            return

        for param_name, sweep in all_results.items():
            vals = [float(k) for k in sweep.keys()]
            accs = [v['avg_accuracy'] for v in sweep.values()]
            bwts = [v['bwt'] for v in sweep.values()]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            ax1.plot(vals, accs, 'o-', linewidth=2, color='steelblue')
            ax1.set_xlabel(param_name)
            ax1.set_ylabel('Average Accuracy')
            ax1.set_title(f'Accuracy vs {param_name}')
            ax1.grid(True, alpha=0.3)

            ax2.plot(vals, bwts, 's-', linewidth=2, color='coral')
            ax2.set_xlabel(param_name)
            ax2.set_ylabel('BWT')
            ax2.set_title(f'Backward Transfer vs {param_name}')
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / f'sensitivity_{param_name}.png', dpi=300)
            plt.close()

        logger.info(f"Sensitivity plots saved to {self.output_dir}")
