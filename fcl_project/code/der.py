import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class DERBuffer:
    """
    Experience Replay Buffer for Dark Experience Replay (DER++).
    Stores samples, labels, and logits (dark knowledge).
    """
    def __init__(self, buffer_size: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.current_index = 0
        self.n_samples = 0
        
        # Buffers (initialized lazily)
        self.images = None
        self.tabular = None
        self.labels = None
        self.logits = None

    def add_data(self, images: torch.Tensor, tabular: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor):
        """Add new samples to the buffer using reservoir sampling or FIFO."""
        batch_size = images.size(0)
        
        # Lazy initialization
        if self.images is None:
            self.images = torch.zeros((self.buffer_size, *images.shape[1:]), device=self.device)
            self.tabular = torch.zeros((self.buffer_size, *tabular.shape[1:]), device=self.device)
            self.labels = torch.zeros((self.buffer_size,), dtype=torch.long, device=self.device)
            self.logits = torch.zeros((self.buffer_size, *logits.shape[1:]), device=self.device)

        for i in range(batch_size):
            # Reservoir Sampling or Simple FIFO
            if self.n_samples < self.buffer_size:
                idx = self.n_samples
            else:
                idx = np.random.randint(0, self.n_samples)
                if idx >= self.buffer_size:
                    self.n_samples += 1
                    continue
            
            self.images[idx] = images[i].detach()
            self.tabular[idx] = tabular[i].detach()
            self.labels[idx] = labels[i].detach()
            self.logits[idx] = logits[i].detach()
            
            if self.n_samples < self.buffer_size:
                self.n_samples += 1

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch from the buffer."""
        if self.n_samples == 0:
            return None
            
        actual_batch_size = min(batch_size, self.n_samples)
        indices = np.random.choice(self.n_samples, actual_batch_size, replace=False)
        
        return (
            self.images[indices],
            self.tabular[indices],
            self.labels[indices],
            self.logits[indices]
        )

def der_loss(logits: torch.Tensor, labels: torch.Tensor, 
             buffer_logits: torch.Tensor, buffer_labels: torch.Tensor,
             alpha: float = 0.1, beta: float = 0.5) -> torch.Tensor:
    """
    DER++ Loss function.
    L = L_task + alpha * MSE(logits, buffer_logits) + beta * CE(logits, buffer_labels)
    """
    # Standard task loss (handled outside or here)
    # This function focus on the regularization terms
    
    # Dark knowledge loss (MSE between current and past logits)
    loss_mse = F.mse_loss(logits, buffer_logits)
    
    # Replay loss (CrossEntropy with past labels)
    loss_ce = F.cross_entropy(logits, buffer_labels)
    
    return alpha * loss_mse + beta * loss_ce
