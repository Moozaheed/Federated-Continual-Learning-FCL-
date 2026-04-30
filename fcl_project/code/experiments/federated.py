"""Federated learning server implementations: FedAvg and FedProx.

FedAvg: McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks"
FedProx: Li et al. (2020) "Federated Optimization in Heterogeneous Networks"
"""

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


class FedAvgServer:
    """FedAvg aggregation server."""

    def __init__(self, global_model: nn.Module, device: str = 'cpu'):
        self.global_model = global_model.to(device)
        self.device = device

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.global_model.state_dict().items()}

    def aggregate(
        self,
        client_state_dicts: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
    ):
        """Weighted average of client parameters (FedAvg)."""
        if client_weights is None:
            client_weights = [1.0 / len(client_state_dicts)] * len(client_state_dicts)

        total = sum(client_weights)
        weights = [w / total for w in client_weights]

        avg_state = {}
        for key in client_state_dicts[0]:
            avg_state[key] = sum(
                w * sd[key].float().to(self.device)
                for w, sd in zip(weights, client_state_dicts)
            ).to(client_state_dicts[0][key].dtype)

        self.global_model.load_state_dict(avg_state)

    def distribute(self, client_models: List[nn.Module]):
        """Copy global parameters to all client models."""
        gsd = self.global_model.state_dict()
        for cm in client_models:
            cm.load_state_dict(copy.deepcopy(gsd))


class FedProxServer(FedAvgServer):
    """FedProx extends FedAvg with a proximal regularization term."""

    def __init__(self, global_model: nn.Module, mu: float = 0.01, device: str = 'cpu'):
        super().__init__(global_model, device)
        self.mu = mu


def fedprox_local_loss(
    model: nn.Module,
    global_params: Dict[str, torch.Tensor],
    mu: float,
) -> torch.Tensor:
    """Proximal term: mu/2 * ||w - w_global||^2."""
    prox = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in global_params:
            prox = prox + ((param - global_params[name].to(param.device)) ** 2).sum()
    return (mu / 2.0) * prox


def train_client_local(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    local_epochs: int,
    task_id: int = 0,
    task_type: str = 'single_label',
    global_params: Optional[Dict[str, torch.Tensor]] = None,
    mu: float = 0.0,
    scheduler=None,
    cl_strategy=None,
) -> Dict[str, torch.Tensor]:
    """Train one federated client locally, return state_dict.

    Supports:
      - FedProx proximal term (when mu > 0 and global_params provided)
      - CL strategy loss augmentation (when cl_strategy provided)
      - LR scheduling (when scheduler provided)
      - Multi-label tasks (when task_type == 'multi_label')
    """
    model.train()
    for epoch in range(local_epochs):
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0].to(device), batch[1].to(device)
            else:
                images = batch['image'].to(device)
                if task_type == 'multi_label':
                    labels = batch['labels'].to(device).float()
                else:
                    key = 'label' if 'label' in batch else 'labels'
                    labels = batch[key].to(device)

            optimizer.zero_grad()
            logits = model(images, task_id=task_id)

            if task_type != 'multi_label' and labels.dim() > 1:
                labels = labels.argmax(dim=1)

            loss = criterion(logits, labels)

            if cl_strategy is not None:
                loss = loss + cl_strategy.compute_loss(
                    model, logits, labels, task_id, task_type=task_type
                )

            if mu > 0 and global_params is not None:
                loss = loss + fedprox_local_loss(model, global_params, mu)

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def create_dirichlet_splits(
    dataset: Dataset,
    n_clients: int,
    alpha: float,
    seed: int = 42,
    task_type: str = 'single_label',
) -> List[Subset]:
    """Non-IID federated splits using Dirichlet distribution.

    Lower alpha -> more heterogeneous. alpha -> inf gives IID.
    For multi-label datasets, uses argmax of multi-hot vector as pseudo-class.
    """
    rng = np.random.RandomState(seed)
    n_data = len(dataset)

    if hasattr(dataset, 'labels') and dataset.labels is not None:
        raw_labels = np.asarray(dataset.labels)
        if raw_labels.ndim > 1 and raw_labels.shape[1] > 1:
            labels = raw_labels.argmax(axis=1)
        else:
            labels = raw_labels.flatten()
    else:
        sample = dataset[0]
        if task_type == 'multi_label' and 'labels' in sample:
            key = 'labels'
        else:
            key = 'label' if 'label' in sample else 'labels'
        raw = [dataset[i][key] for i in range(n_data)]
        labels = np.array([
            l.item() if isinstance(l, torch.Tensor) and l.dim() == 0
            else (l.argmax().item() if isinstance(l, torch.Tensor) and l.dim() > 0 else int(l))
            for l in raw
        ])

    n_classes = len(np.unique(labels))
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet([alpha] * n_clients)
        proportions = (proportions * len(idx_c)).astype(int)
        proportions[-1] = len(idx_c) - proportions[:-1].sum()

        start = 0
        for k in range(n_clients):
            client_indices[k].extend(idx_c[start:start + proportions[k]].tolist())
            start += proportions[k]

    for ci in client_indices:
        rng.shuffle(ci)

    logger.info(
        f"Dirichlet splits (alpha={alpha}): sizes={[len(c) for c in client_indices]}"
    )
    return [Subset(dataset, ci) for ci in client_indices]
