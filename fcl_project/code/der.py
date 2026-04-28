import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict

class DERBuffer:
    """
    Experience Replay Buffer for Dark Experience Replay (DER++).
    
    Stores multimodal samples (images + tabular data), labels, and logits
    (dark knowledge from previous models) for continual learning with
    knowledge distillation.
    
    Key Features:
    - Reservoir sampling for memory-efficient storage
    - Supports multimodal data (images and tabular)
    - Dark knowledge preservation via logit storage
    - Comprehensive input validation and error handling
    - Dynamic batch size adaptation
    
    References:
    - Buzzega et al. (2020): DER - Dark Experience Replay
    - Buzzega et al. (2021): DER++ with improved sampling
    
    Buffer Capacity: buffer_size samples
    Storage: O(buffer_size * (image_size + tabular_dim + num_classes))
    """
    
    def __init__(self, buffer_size: int, device: str = "cpu"):
        """
        Initialize DER++ experience replay buffer.
        
        Args:
            buffer_size: Maximum number of samples to store
                - Typical range: 1000-10000 for medical applications
                - Larger buffer → better coverage, more memory
                - Smaller buffer → faster sampling, lower memory
            device: Device to store buffers on ("cuda" or "cpu")
        
        Raises:
            ValueError: If buffer_size <= 0
        
        Examples:
            >>> buffer = DERBuffer(buffer_size=5000, device="cuda")
            >>> print(buffer.capacity, buffer.n_samples)
            5000, 0
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        
        self.buffer_size = buffer_size
        self.device = device
        self.current_index = 0
        self.n_samples = 0
        
        # Buffers (initialized lazily on first add_data call)
        self.images = None
        self.tabular = None
        self.labels = None
        self.logits = None
        
        # Metadata for validation
        self._initialized = False
        self._image_shape = None
        self._tabular_dim = None
        self._num_classes = None
    
    @property
    def capacity(self) -> int:
        """Total buffer capacity."""
        return self.buffer_size
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self.n_samples >= self.buffer_size
    
    @property
    def occupancy_ratio(self) -> float:
        """Percentage of buffer filled (0.0 to 1.0)."""
        return self.n_samples / self.buffer_size if self.buffer_size > 0 else 0.0
    
    def add_data(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor
    ) -> None:
        """
        Add new multimodal samples to the experience replay buffer.
        
        Uses reservoir sampling to maintain uniform distribution across all
        seen data when buffer is full. Validates all inputs for consistency.
        
        Args:
            images: (batch_size, 3, H, W) - RGB images [0, 1] normalized
            tabular: (batch_size, num_features) - clinical tabular features
            labels: (batch_size,) - ground truth labels [0, num_classes-1]
            logits: (batch_size, num_classes) - model predictions / dark knowledge
        
        Raises:
            ValueError: If input shapes are inconsistent or batch sizes don't match
            TypeError: If inputs are not torch tensors
            RuntimeError: If device placement fails
        
        Examples:
            >>> buffer = DERBuffer(buffer_size=1000, device="cuda")
            >>> images = torch.randn(32, 3, 224, 224).cuda()
            >>> tabular = torch.randn(32, 13).cuda()
            >>> labels = torch.randint(0, 2, (32,)).cuda()
            >>> logits = torch.randn(32, 2).cuda()
            >>> buffer.add_data(images, tabular, labels, logits)
            >>> print(f"Buffer now has {buffer.n_samples} samples")
            Buffer now has 32 samples
        """
        # Input validation
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"images must be torch.Tensor, got {type(images)}")
        if not isinstance(tabular, torch.Tensor):
            raise TypeError(f"tabular must be torch.Tensor, got {type(tabular)}")
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"labels must be torch.Tensor, got {type(labels)}")
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"logits must be torch.Tensor, got {type(logits)}")
        
        batch_size = images.size(0)
        
        # Batch size consistency check
        if tabular.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch: images={batch_size}, tabular={tabular.size(0)}")
        if labels.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch: images={batch_size}, labels={labels.size(0)}")
        if logits.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch: images={batch_size}, logits={logits.size(0)}")
        
        # Image shape validation
        if images.dim() != 4:
            raise ValueError(f"images must be 4D (batch, channels, H, W), got shape {images.shape}")
        if images.size(1) != 3:
            raise ValueError(f"images must have 3 channels (RGB), got {images.size(1)}")
        
        # First call: initialize buffers
        if not self._initialized:
            self._initialize_buffers(images, tabular, labels, logits)
        else:
            # Subsequent calls: validate consistency
            self._validate_shapes(images, tabular, labels, logits)
        
        # Move data to device
        try:
            images = images.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)
            logits = logits.to(self.device)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to move data to device {self.device}: {e}")
        
        # Add samples using reservoir sampling
        for i in range(batch_size):
            if self.n_samples < self.buffer_size:
                # Buffer not full: add to next available slot
                idx = self.n_samples
            else:
                # Buffer full: use reservoir sampling
                # Probability of replacing element: 1 / (n_samples + 1)
                idx = np.random.randint(0, self.n_samples)
                if idx >= self.buffer_size:
                    # This sample shouldn't be stored (probability check failed)
                    self.n_samples += 1
                    continue
            
            # Store sample components
            self.images[idx] = images[i].detach()
            self.tabular[idx] = tabular[i].detach()
            self.labels[idx] = labels[i].detach()
            self.logits[idx] = logits[i].detach()
            
            # Increment counter only once
            if self.n_samples < self.buffer_size:
                self.n_samples += 1
    
    def _initialize_buffers(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor
    ) -> None:
        """Initialize buffer tensors with correct shapes on first call."""
        self._image_shape = images.shape[1:]
        self._tabular_dim = tabular.size(1)
        self._num_classes = logits.size(1)
        
        # Pre-allocate buffer tensors
        self.images = torch.zeros(
            (self.buffer_size, *self._image_shape),
            device=self.device,
            dtype=images.dtype
        )
        self.tabular = torch.zeros(
            (self.buffer_size, self._tabular_dim),
            device=self.device,
            dtype=tabular.dtype
        )
        self.labels = torch.zeros(
            (self.buffer_size,),
            device=self.device,
            dtype=labels.dtype
        )
        self.logits = torch.zeros(
            (self.buffer_size, self._num_classes),
            device=self.device,
            dtype=logits.dtype
        )
        
        self._initialized = True
    
    def _validate_shapes(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor
    ) -> None:
        """Validate that new data matches buffer schema."""
        if images.shape[1:] != self._image_shape:
            raise ValueError(
                f"Image shape mismatch: expected {self._image_shape}, "
                f"got {images.shape[1:]}"
            )
        if tabular.size(1) != self._tabular_dim:
            raise ValueError(
                f"Tabular dimension mismatch: expected {self._tabular_dim}, "
                f"got {tabular.size(1)}"
            )
        if logits.size(1) != self._num_classes:
            raise ValueError(
                f"Number of classes mismatch: expected {self._num_classes}, "
                f"got {logits.size(1)}"
            )
    
    def sample_batch(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Sample a random batch from the buffer.
        
        Performs uniform sampling with replacement. If buffer has fewer
        samples than requested, returns all available samples.
        
        Args:
            batch_size: Number of samples to retrieve
        
        Returns:
            Dictionary with keys:
                - 'images': (actual_batch_size, 3, H, W)
                - 'tabular': (actual_batch_size, num_features)
                - 'labels': (actual_batch_size,)
                - 'logits': (actual_batch_size, num_classes)
            Or None if buffer is empty
        
        Raises:
            ValueError: If batch_size <= 0
        
        Examples:
            >>> batch = buffer.sample_batch(batch_size=32)
            >>> if batch is not None:
            ...     images, tabular = batch['images'], batch['tabular']
            ...     print(images.shape)
            torch.Size([32, 3, 224, 224])
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        # Check if buffer has data
        if self.n_samples == 0:
            return None
        
        # Determine actual batch size (may be smaller than requested)
        actual_batch_size = min(batch_size, self.n_samples)
        
        # Uniform sampling with replacement
        indices = np.random.choice(
            self.n_samples,
            size=actual_batch_size,
            replace=(actual_batch_size > self.n_samples)
        )
        
        # Validate indices are within bounds
        if np.any(indices >= self.buffer_size):
            indices = indices % self.buffer_size
        
        # Return batch dictionary
        return {
            'images': self.images[indices],
            'tabular': self.tabular[indices],
            'labels': self.labels[indices],
            'logits': self.logits[indices]
        }
    
    def clear(self) -> None:
        """Clear buffer and reset counter."""
        if self.images is not None:
            self.images.zero_()
        if self.tabular is not None:
            self.tabular.zero_()
        if self.labels is not None:
            self.labels.zero_()
        if self.logits is not None:
            self.logits.zero_()
        
        self.n_samples = 0
        self.current_index = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics for monitoring."""
        return {
            'capacity': self.buffer_size,
            'n_samples': self.n_samples,
            'occupancy_ratio': self.occupancy_ratio,
            'is_full': self.is_full
        }


def der_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    buffer_logits: torch.Tensor,
    buffer_labels: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7
) -> torch.Tensor:
    """
    DER++ Loss function for continual learning.
    
    Combines three components:
    1. Standard task loss: CE(logits, labels) on new data
    2. Dark knowledge loss: MSE(logits, buffer_logits) for knowledge distillation
    3. Replay loss: CE(logits, buffer_labels) for task rehearsal
    
    Total Loss: L_task + alpha * MSE(logits, buffer_logits) + beta * CE(logits, buffer_labels)
    
    Args:
        logits: (batch_size, num_classes) - predictions on new data
        labels: (batch_size,) - labels on new data
        buffer_logits: (batch_size, num_classes) - stored logits from old models
        buffer_labels: (batch_size,) - stored labels from replay buffer
        alpha: Weight for dark knowledge loss (default: 0.3)
        beta: Weight for replay loss (default: 0.7)
    
    Returns:
        loss: Scalar tensor
    
    Examples:
        >>> logits_new = model(x_new)
        >>> batch = buffer.sample_batch(32)
        >>> loss = der_loss(
        ...     logits_new, y_new,
        ...     batch['logits'], batch['labels'],
        ...     alpha=0.3, beta=0.7
        ... )
    """
    # Standard task loss (handled outside or via cross_entropy on new data)
    
    # Dark knowledge loss: minimize difference from previous model outputs
    # MSE is more stable for logit matching than KL divergence for DER++
    loss_mse = F.mse_loss(logits, buffer_logits, reduction='mean')
    
    # Replay loss: maintain performance on previous tasks
    # Cross-entropy ensures correct classification on replayed samples
    loss_ce = F.cross_entropy(logits, buffer_labels, reduction='mean')
    
    # Combined DER++ loss
    total_loss = alpha * loss_mse + beta * loss_ce
    
    return total_loss
