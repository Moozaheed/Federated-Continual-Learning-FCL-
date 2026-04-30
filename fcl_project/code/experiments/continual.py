"""Continual learning strategy wrappers.

Provides a unified interface for:
  - FineTune (no protection, baseline)
  - EWC (Kirkpatrick et al. 2017)
  - DER++ at feature level (Buzzega et al. 2021)
  - Generative Replay with feature-space VAE
"""

import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch extraction helper
# ---------------------------------------------------------------------------

def _extract_cl_batch(batch, device: str, task_type: str = 'single_label'):
    """Extract images and labels from a batch, respecting task_type."""
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device), batch[1].to(device)

    images = batch['image'].to(device)
    if task_type == 'multi_label':
        labels = batch['labels'].to(device).float()
    else:
        key = 'label' if 'label' in batch else 'labels'
        labels = batch[key].to(device)
    return images, labels


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ContinualStrategy(ABC):
    @abstractmethod
    def pre_task(self, model: nn.Module, task_id: int, **kwargs):
        """Called before training on a new task."""

    @abstractmethod
    def compute_loss(
        self, model: nn.Module, logits: torch.Tensor,
        labels: torch.Tensor, task_id: int, **kwargs
    ) -> torch.Tensor:
        """Return additional CL loss (added to main loss). Zero if none."""

    @abstractmethod
    def post_task(
        self, model: nn.Module, task_id: int,
        train_loader: DataLoader, **kwargs
    ):
        """Called after training on a task (e.g., compute Fisher, fill buffer)."""


# ---------------------------------------------------------------------------
# Fine-tune baseline
# ---------------------------------------------------------------------------

class FineTuneStrategy(ContinualStrategy):
    """No CL protection — pure sequential fine-tuning."""

    def pre_task(self, model, task_id, **kw):
        pass

    def compute_loss(self, model, logits, labels, task_id, **kw):
        return torch.tensor(0.0, device=logits.device)

    def post_task(self, model, task_id, train_loader, **kw):
        pass


# ---------------------------------------------------------------------------
# EWC
# ---------------------------------------------------------------------------

class EWCStrategy(ContinualStrategy):
    """Elastic Weight Consolidation (Kirkpatrick et al. 2017)."""

    def __init__(self, lambda_ewc: float = 0.5, device: str = 'cpu'):
        self.lambda_ewc = lambda_ewc
        self.device = device
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def pre_task(self, model, task_id, **kw):
        pass

    def compute_loss(self, model, logits, labels, task_id, **kw):
        if not self.fisher:
            return torch.tensor(0.0, device=logits.device)
        ewc_loss = torch.tensor(0.0, device=logits.device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                ewc_loss = ewc_loss + (
                    self.fisher[name].to(param.device) *
                    (param - self.optimal_params[name].to(param.device)) ** 2
                ).sum()
        return self.lambda_ewc * ewc_loss

    def post_task(self, model, task_id, train_loader, **kw):
        """Compute diagonal Fisher and store optimal params."""
        task_type = kw.get('task_type', 'single_label')
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        n_samples = 0

        for batch in train_loader:
            images, labels = _extract_cl_batch(batch, self.device, task_type)

            model.zero_grad()
            logits = model(images, task_id=task_id)

            if task_type == 'multi_label':
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            else:
                if labels.dim() > 1:
                    labels = labels.argmax(dim=1)
                loss = F.cross_entropy(logits, labels)
            loss.backward()

            for n, p in model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.data ** 2
            n_samples += labels.size(0)

        for n in fisher:
            fisher[n] /= max(n_samples, 1)

        if self.fisher:
            for n in fisher:
                if n in self.fisher:
                    fisher[n] = fisher[n] + self.fisher[n].to(fisher[n].device)

        self.fisher = {n: f.detach().cpu() for n, f in fisher.items()}
        self.optimal_params = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        logger.info(f"EWC: Fisher computed for task {task_id}")


# ---------------------------------------------------------------------------
# Feature-level DER++ buffer
# ---------------------------------------------------------------------------

class FeatureDERBuffer:
    """Lightweight replay buffer storing extracted features, not raw images.

    Labels are always stored as single-label integers (multi-label tasks
    are reduced via argmax before storage). The primary DER++ signal is
    logit matching, so this is a minor approximation.
    """

    def __init__(self, buffer_size: int = 5000, device: str = 'cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.n_samples = 0
        self.features: Optional[torch.Tensor] = None
        self.logits: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self._task_ids: Optional[torch.Tensor] = None

    def add(
        self, features: torch.Tensor, logits: torch.Tensor,
        labels: torch.Tensor, task_ids: torch.Tensor,
    ):
        feat = features.detach().cpu()
        log = logits.detach().cpu()
        lab = labels.detach().cpu()
        tid = task_ids.detach().cpu()

        if lab.dim() > 1:
            lab = lab.argmax(dim=1)

        if self.features is None:
            self.features = torch.zeros(self.buffer_size, feat.size(1))
            self.logits = torch.zeros(self.buffer_size, log.size(1))
            self.labels = torch.zeros(self.buffer_size, dtype=torch.long)
            self._task_ids = torch.zeros(self.buffer_size, dtype=torch.long)

        if log.size(1) > self.logits.size(1):
            old = self.logits
            self.logits = torch.zeros(self.buffer_size, log.size(1))
            self.logits[:, :old.size(1)] = old

        for i in range(feat.size(0)):
            if self.n_samples < self.buffer_size:
                idx = self.n_samples
            else:
                j = np.random.randint(0, self.n_samples + 1)
                if j >= self.buffer_size:
                    self.n_samples += 1
                    continue
                idx = j
            self.features[idx] = feat[i]
            self.logits[idx] = 0
            self.logits[idx, :log.size(1)] = log[i]
            self.labels[idx] = lab[i]
            self._task_ids[idx] = tid[i]
            self.n_samples += 1

    def sample(self, batch_size: int):
        stored = min(self.n_samples, self.buffer_size)
        if stored == 0:
            return None
        idx = np.random.choice(stored, size=min(batch_size, stored), replace=False)
        return {
            'features': self.features[idx],
            'logits': self.logits[idx],
            'labels': self.labels[idx],
            'task_ids': self._task_ids[idx],
        }


class DERStrategy(ContinualStrategy):
    """Feature-level Dark Experience Replay++ (Buzzega et al. 2021).

    Stores 576-dim MobileNetV3 features instead of raw 224x224 images,
    reducing buffer memory from ~3 GB to ~12 MB for 5000 samples.
    """

    def __init__(
        self, buffer_size: int = 5000,
        alpha: float = 0.3, beta: float = 0.7,
        device: str = 'cpu',
    ):
        self.buffer = FeatureDERBuffer(buffer_size, device)
        self.alpha = alpha
        self.beta = beta
        self.device = device

    def pre_task(self, model, task_id, **kw):
        pass

    def compute_loss(self, model, logits, labels, task_id, **kw):
        """Replay loss at feature level."""
        batch = self.buffer.sample(logits.size(0))
        if batch is None:
            return torch.tensor(0.0, device=logits.device)

        buf_feat = batch['features'].to(self.device)
        buf_logits = batch['logits'].to(self.device)
        buf_labels = batch['labels'].to(self.device).long()
        buf_tids = batch['task_ids'].to(self.device)

        unique_tids = buf_tids.unique()
        replay_loss = torch.tensor(0.0, device=self.device)
        count = 0

        for tid in unique_tids:
            mask = buf_tids == tid
            if not mask.any():
                continue
            head_key = f'task_{tid.item()}'
            if not hasattr(model, 'heads') or head_key not in model.heads:
                continue
            head = model.heads[head_key]
            new_logits = head(buf_feat[mask])
            n_cls = new_logits.size(1)

            old_log = buf_logits[mask][:, :n_cls]
            loss_mse = F.mse_loss(new_logits, old_log)
            loss_ce = F.cross_entropy(new_logits, buf_labels[mask])
            replay_loss = replay_loss + self.alpha * loss_mse + self.beta * loss_ce
            count += 1

        return replay_loss / max(count, 1)

    def post_task(self, model, task_id, train_loader, **kw):
        """Extract features from current task and store in buffer."""
        task_type = kw.get('task_type', 'single_label')
        model.eval()
        max_logit_dim = max(
            model.heads[k][-1].out_features for k in model.heads
        ) if hasattr(model, 'heads') else 2

        with torch.no_grad():
            for batch in train_loader:
                images, labels = _extract_cl_batch(batch, self.device, task_type)

                features = model.extract_features(images)
                logits = model(images, task_id=task_id)

                if task_type != 'multi_label' and labels.dim() > 1:
                    labels = labels.argmax(dim=1)

                padded = torch.zeros(logits.size(0), max_logit_dim, device=self.device)
                padded[:, :logits.size(1)] = logits

                tid_vec = torch.full((labels.size(0),), task_id, dtype=torch.long, device=self.device)
                self.buffer.add(features, padded, labels, tid_vec)

        logger.info(
            f"DER++: buffer filled for task {task_id}, "
            f"total={self.buffer.n_samples}"
        )


# ---------------------------------------------------------------------------
# Generative Replay (feature-space VAE)
# ---------------------------------------------------------------------------

class _FeatureVAE(nn.Module):
    """Simple VAE operating on extracted feature vectors."""

    def __init__(self, input_dim: int = 576, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class GenReplayStrategy(ContinualStrategy):
    """Generative Replay using a feature-space VAE per task."""

    def __init__(
        self, feature_dim: int = 576, latent_dim: int = 64,
        replay_per_task: int = 500, vae_epochs: int = 30,
        device: str = 'cpu',
    ):
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.replay_per_task = replay_per_task
        self.vae_epochs = vae_epochs
        self.device = device
        self.generators: List[_FeatureVAE] = []
        self.task_n_classes: List[int] = []
        self._replay_features: Optional[torch.Tensor] = None
        self._replay_labels: Optional[torch.Tensor] = None
        self._replay_task_ids: Optional[torch.Tensor] = None

    def pre_task(self, model, task_id, **kw):
        """Generate synthetic features from previous VAEs for replay."""
        if not self.generators:
            self._replay_features = None
            return

        all_feat, all_lab, all_tid = [], [], []
        for prev_tid, (vae, n_cls) in enumerate(zip(self.generators, self.task_n_classes)):
            vae.eval()
            with torch.no_grad():
                z = torch.randn(self.replay_per_task, self.latent_dim, device=self.device)
                synth = vae.decode(z)
            all_feat.append(synth.cpu())
            all_lab.append(torch.randint(0, n_cls, (self.replay_per_task,)))
            all_tid.append(torch.full((self.replay_per_task,), prev_tid, dtype=torch.long))

        self._replay_features = torch.cat(all_feat)
        self._replay_labels = torch.cat(all_lab)
        self._replay_task_ids = torch.cat(all_tid)

    def compute_loss(self, model, logits, labels, task_id, **kw):
        if self._replay_features is None:
            return torch.tensor(0.0, device=logits.device)

        replay_loss = torch.tensor(0.0, device=logits.device)
        feat = self._replay_features.to(self.device)
        lab = self._replay_labels.to(self.device)
        tids = self._replay_task_ids.to(self.device)
        count = 0

        for tid_val in tids.unique():
            mask = tids == tid_val
            head_key = f'task_{tid_val.item()}'
            if not hasattr(model, 'heads') or head_key not in model.heads:
                continue
            head = model.heads[head_key]
            out = head(feat[mask])
            replay_loss = replay_loss + F.cross_entropy(out, lab[mask])
            count += 1

        return replay_loss / max(count, 1)

    def post_task(self, model, task_id, train_loader, **kw):
        """Train a VAE on extracted features from this task."""
        task_type = kw.get('task_type', 'single_label')
        model.eval()
        all_features = []
        n_classes = 0

        with torch.no_grad():
            for batch in train_loader:
                images, labels = _extract_cl_batch(batch, self.device, task_type)

                if task_type == 'multi_label':
                    n_classes = max(n_classes, labels.size(-1))
                else:
                    if labels.dim() > 1:
                        labels = labels.argmax(dim=1)
                    n_classes = max(n_classes, labels.max().item() + 1)
                feat = model.extract_features(images)
                all_features.append(feat.cpu())

        all_features = torch.cat(all_features)
        self.task_n_classes.append(n_classes)

        vae = _FeatureVAE(self.feature_dim, self.latent_dim).to(self.device)
        opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

        vae.train()
        for _ in range(self.vae_epochs):
            perm = torch.randperm(all_features.size(0))
            for start in range(0, all_features.size(0), 128):
                batch_f = all_features[perm[start:start + 128]].to(self.device)
                recon, mu, logvar = vae(batch_f)
                recon_loss = F.mse_loss(recon, batch_f)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl
                opt.zero_grad()
                loss.backward()
                opt.step()

        vae.eval()
        self.generators.append(vae)
        logger.info(f"GenReplay: VAE trained for task {task_id}")
