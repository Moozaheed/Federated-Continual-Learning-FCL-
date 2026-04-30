"""Accuracy matrix and continual learning metrics.

Implements the standard CL evaluation protocol from
Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for Continual Learning".
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score


class AccuracyMatrix:
    """Track R[i][j] = performance on task j after training through task i.

    Standard evaluation for continual learning. All CL metrics
    (BWT, FWT, forgetting, average accuracy) derive from this matrix.
    For single-label tasks, values are accuracy; for multi-label, mean AUC-ROC.
    """

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        self.matrix = np.zeros((n_tasks, n_tasks))

    def update(self, trained_through_task: int, eval_task: int, accuracy: float):
        self.matrix[trained_through_task, eval_task] = accuracy

    def bwt(self) -> float:
        """Backward Transfer: how learning new tasks affects old tasks."""
        T = self.n_tasks
        if T < 2:
            return 0.0
        return sum(
            self.matrix[T - 1, i] - self.matrix[i, i] for i in range(T - 1)
        ) / (T - 1)

    def fwt(self, baselines: Optional[np.ndarray] = None) -> float:
        """Forward Transfer: how old tasks help new tasks."""
        T = self.n_tasks
        if T < 2:
            return 0.0
        if baselines is None:
            baselines = np.zeros(T)
        return sum(
            self.matrix[i - 1, i] - baselines[i] for i in range(1, T)
        ) / (T - 1)

    def average_accuracy(self) -> float:
        """Average accuracy across all tasks after final training."""
        T = self.n_tasks
        return float(np.mean(self.matrix[T - 1, :T]))

    def forgetting(self) -> float:
        """Average forgetting: max accuracy on each task minus final accuracy."""
        T = self.n_tasks
        if T < 2:
            return 0.0
        forg = []
        for j in range(T - 1):
            max_acc = max(self.matrix[i, j] for i in range(j, T))
            forg.append(max_acc - self.matrix[T - 1, j])
        return float(np.mean(forg))

    def to_dict(self) -> Dict:
        return {
            'matrix': self.matrix.tolist(),
            'bwt': self.bwt(),
            'fwt': self.fwt(),
            'average_accuracy': self.average_accuracy(),
            'forgetting': self.forgetting(),
        }


def _extract_batch(
    batch, device: str, task_type: str = 'single_label',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified batch extraction for both single-label and multi-label tasks."""
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device), batch[1].to(device)

    images = batch['image'].to(device)
    if task_type == 'multi_label':
        labels = batch['labels'].to(device).float()
    else:
        key = 'label' if 'label' in batch else 'labels'
        labels = batch[key].to(device)
    return images, labels


def evaluate_all_tasks(
    model: nn.Module,
    task_test_loaders: List[DataLoader],
    device: str,
    task_ids: Optional[List[int]] = None,
    task_types: Optional[List[str]] = None,
) -> List[float]:
    """Evaluate model on each task's test set.

    For single-label tasks, returns accuracy.
    For multi-label tasks, returns mean AUC-ROC across findings.
    """
    model.eval()
    if task_ids is None:
        task_ids = list(range(len(task_test_loaders)))
    if task_types is None:
        task_types = ['single_label'] * len(task_test_loaders)

    scores = []
    with torch.no_grad():
        for tid, loader, ttype in zip(task_ids, task_test_loaders, task_types):
            if ttype == 'multi_label':
                scores.append(_evaluate_multi_label(model, loader, tid, device))
            else:
                scores.append(_evaluate_single_label(model, loader, tid, device))
    return scores


def _evaluate_single_label(
    model: nn.Module, loader: DataLoader, tid: int, device: str,
) -> float:
    correct = 0
    total = 0
    for batch in loader:
        images, labels = _extract_batch(batch, device, 'single_label')
        logits = model(images, task_id=tid)
        preds = logits.argmax(dim=1)
        if labels.dim() > 1:
            labels = labels.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def _evaluate_multi_label(
    model: nn.Module, loader: DataLoader, tid: int, device: str,
) -> float:
    all_probs = []
    all_labels = []
    for batch in loader:
        images, labels = _extract_batch(batch, device, 'multi_label')
        logits = model(images, task_id=tid)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    try:
        aucs = []
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                aucs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
        return float(np.mean(aucs)) if aucs else 0.0
    except ValueError:
        return 0.0


def compute_roc_auc_per_task(
    model: nn.Module,
    task_test_loaders: List[DataLoader],
    device: str,
    task_ids: Optional[List[int]] = None,
    task_types: Optional[List[str]] = None,
) -> List[float]:
    """Compute ROC-AUC for each task."""
    model.eval()
    if task_ids is None:
        task_ids = list(range(len(task_test_loaders)))
    if task_types is None:
        task_types = ['single_label'] * len(task_test_loaders)

    auc_scores = []
    with torch.no_grad():
        for tid, loader, ttype in zip(task_ids, task_test_loaders, task_types):
            if ttype == 'multi_label':
                auc_scores.append(_evaluate_multi_label(model, loader, tid, device))
            else:
                auc_scores.append(
                    _compute_single_label_auc(model, loader, tid, device)
                )
    return auc_scores


def _compute_single_label_auc(
    model: nn.Module, loader: DataLoader, tid: int, device: str,
) -> float:
    all_probs = []
    all_labels = []
    for batch in loader:
        images, labels = _extract_batch(batch, device, 'single_label')
        logits = model(images, task_id=tid)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        if labels.dim() > 1:
            labels = labels.argmax(dim=1)
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    try:
        return float(roc_auc_score(
            all_labels, all_probs, multi_class='ovr', average='macro'
        ))
    except ValueError:
        return 0.0
