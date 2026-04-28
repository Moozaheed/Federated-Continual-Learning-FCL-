"""
FCL Configuration Module
Centralized configuration for Federated Continual Learning experiments.
"""

import torch
from typing import Dict, List, Tuple

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================

class ModelConfig:
    """FT-Transformer architecture configuration."""
    
    # Input features
    input_dim: int = 13  # Clinical features (UCI Heart Disease)
    
    # Feature tokenization
    token_dim: int = 192  # Increased for advanced run (from 64)
    n_feature_tokens: int = input_dim  # One token per feature
    
    # Prompt tuning
    n_prompt_tokens: int = 10  # Increased for advanced run (from 5)
    prompt_init_std: float = 0.02
    
    # Transformer blocks
    n_transformer_blocks: int = 6  # Increased for advanced run (from 3)
    n_attention_heads: int = 8
    attention_dropout: float = 0.1
    mlp_hidden_factor: int = 4  # Increased for advanced run (from 2)
    mlp_dropout: float = 0.1
    
    # Output
    output_dim: int = 2  # Binary classification (presence/absence of disease)

class MultimodalConfig:
    """Configuration for Multimodal branch (Image + Tabular)."""
    enabled: bool = True
    image_size: Tuple[int, int] = (224, 224)
    image_channels: int = 3
    cnn_backbone: str = "mobilenet_v3_small"
    feature_dim: int = 576  # Output of MobileNetV3-Small GAP
    fusion_strategy: str = "concat"  # "concat" or "attention"
    hidden_dim: int = 256  # Dimension after fusion

class DERConfig:
    """
    Dark Experience Replay (DER++) Configuration.
    
    DER++ combines experience replay with knowledge distillation (logit matching)
    to prevent catastrophic forgetting in continual learning scenarios.
    
    Reference: "Dark Experience for General Continual Learning" (Buzzega et al., 2021)
    
    Attributes:
        enabled: Whether to use DER++ in training pipeline
        buffer_size: Maximum samples stored in replay buffer (memory constraint)
        alpha: Weight for logit matching loss (dark knowledge from past tasks)
               Range: [0.0, 1.0], higher = more weight on replay logits
        beta: Weight for standard cross-entropy loss on replay samples
              Range: [0.0, 1.0], typically alpha + beta ≤ 1.0
        batch_size: Number of samples to replay per batch (trade-off: memory vs stability)
        use_logits: Whether to store and use logits (dark knowledge) from past models
        sampling_strategy: "reservoir" (uniform) or "recent" (biased towards recent samples)
        use_biased_weights: Whether to apply importance weighting to replay samples
    """
    enabled: bool = True
    buffer_size: int = 2000
    alpha: float = 0.3      # Weight for logit matching (dark knowledge)
    beta: float = 0.7       # Weight for standard replay (labels)
    batch_size: int = 32
    use_logits: bool = True # Store and use logits for knowledge distillation
    sampling_strategy: str = "reservoir"  # "reservoir" or "recent"
    use_biased_weights: bool = False  # For future importance weighting
    
    def validate(self) -> bool:
        """
        Validate DER++ configuration parameters.
        
        Returns:
            bool: True if valid, raises ValueError otherwise
        
        Raises:
            ValueError: If any parameter is invalid
        """
        errors: List[str] = []
        
        # Validate buffer size
        if self.buffer_size <= 0:
            errors.append(f"buffer_size must be > 0, got {self.buffer_size}")
        if self.buffer_size > 10000:
            errors.append(f"buffer_size > 10000 may cause memory issues")
        
        # Validate loss weights
        if not (0.0 <= self.alpha <= 1.0):
            errors.append(f"alpha must be in [0.0, 1.0], got {self.alpha}")
        if not (0.0 <= self.beta <= 1.0):
            errors.append(f"beta must be in [0.0, 1.0], got {self.beta}")
        
        # Validate batch size
        if self.batch_size <= 0:
            errors.append(f"batch_size must be > 0, got {self.batch_size}")
        if self.batch_size > self.buffer_size:
            errors.append(f"batch_size ({self.batch_size}) should not exceed buffer_size ({self.buffer_size})")
        
        # Validate sampling strategy
        valid_strategies = ["reservoir", "recent"]
        if self.sampling_strategy not in valid_strategies:
            errors.append(f"sampling_strategy must be in {valid_strategies}, got {self.sampling_strategy}")
        
        if errors:
            raise ValueError("DERConfig validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    def get_config_dict(self) -> Dict[str, any]:
        """
        Return configuration as dictionary for logging/tracking.
        
        Returns:
            Dictionary with all DER++ configuration parameters
        """
        return {
            'enabled': self.enabled,
            'buffer_size': self.buffer_size,
            'alpha': self.alpha,
            'beta': self.beta,
            'batch_size': self.batch_size,
            'use_logits': self.use_logits,
            'sampling_strategy': self.sampling_strategy,
            'use_biased_weights': self.use_biased_weights
        }


class TrainingConfig:
    """Training hyperparameters."""
    
    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_type: str = "adam"
    
    # Training schedule
    batch_size: int = 32
    epochs_per_task: int = 10
    early_stopping_patience: int = 5
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.1
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# TRAINING PARAMETERS
# ============================================================================


# ============================================================================
# CONTINUAL LEARNING PARAMETERS
# ============================================================================

class ContinualLearningConfig:
    """Continual learning strategy configuration."""
    
    # Elastic Weight Consolidation (EWC)
    ewc_lambda: float = 0.5  # EWC regularization strength
    fisher_damping: float = 1e-4  # Numerical stability for Fisher matrix
    
    # Prompt Tuning
    prompt_learning_rate: float = 1e-2  # Separate LR for prompt tokens
    prompt_init_std: float = 0.02
    
    # Generative Replay (VAE)
    vae_enabled: bool = True
    replay_buffer_size: int = 100  # Synthetic samples per task
    replay_weight: float = 0.5  # Balance real vs synthetic samples
    
    # VAE Architecture
    vae_input_dim: int = 13  # Same as input_dim
    vae_hidden_dims: List[int] = [8, 4]  # Encoder path: 13 -> 8 -> 4
    vae_latent_dim: int = 4
    vae_learning_rate: float = 1e-3
    vae_kl_weight: float = 1.0  # KL divergence weight
    
    # Task detection
    concept_drift_threshold: float = 0.15  # Threshold for detecting new task
    max_tasks: int = 10  # Maximum number of continual learning tasks


# ============================================================================
# FEDERATED LEARNING PARAMETERS
# ============================================================================

class FederatedConfig:
    """Federated learning orchestration configuration."""
    
    # Flower server setup
    server_address: str = "127.0.0.1:8080"
    num_rounds: int = 20  # Communication rounds
    min_clients: int = 2  # Minimum clients to start round
    min_available_clients: int = 2
    
    # Client configuration
    num_hospitals: int = 4  # Number of federated participants
    hospital_names: List[str] = [
        "Urban_Medical_Center",
        "Rural_Community_Hospital",
        "Specialty_Cardiac_Clinic",
        "University_Teaching_Hospital"
    ]
    
    # Data distribution
    non_iid_factor: float = 0.7  # 1.0 = IID, 0.0 = extremely Non-IID
    
    # Aggregation
    aggregation_strategy: str = "weighted"  # "weighted" or "uniform"
    client_fraction: float = 1.0  # Fraction of clients per round
    
    # Communication efficiency
    compression_enabled: bool = False
    compression_factor: float = 0.5  # % of gradients to send


# ============================================================================
# DIFFERENTIAL PRIVACY PARAMETERS
# ============================================================================

class PrivacyConfig:
    """Differential Privacy configuration."""
    
    # Opacus settings
    dp_enabled: bool = True
    target_epsilon: float = 1.0  # Privacy budget (lower = more private)
    target_delta: float = 1e-5  # Failure probability
    
    # Gradient clipping
    max_grad_norm: float = 1.0  # L2 norm clipping
    
    # Noise addition
    noise_scale: float = 1.0  # Gaussian noise std
    
    # Accounting
    noise_multiplier: float = 1.0  # Derived from epsilon/delta targets


# ============================================================================
# DATASET PARAMETERS
# ============================================================================

class DataConfig:
    """Dataset configuration."""
    
    # UCI Heart Disease
    dataset_name: str = "uci_heart_disease"
    n_samples_per_hospital: int = 50  # Samples per hospital
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Feature normalization
    normalize_features: bool = True
    normalization_type: str = "zscore"  # "zscore" or "minmax"
    
    # MIMIC-IV (future)
    mimic_enabled: bool = False
    mimic_path: str = "/data/mimic-iv"


# ============================================================================
# VISUALIZATION & LOGGING
# ============================================================================

class LoggingConfig:
    """Logging and visualization configuration."""
    
    # Metrics tracking
    log_frequency: int = 100  # Log every N batches
    save_frequency: int = 5  # Save checkpoint every N epochs
    
    # Visualization
    plot_dpi: int = 300
    plot_format: str = "png"
    save_plots: bool = True
    plots_dir: str = "./plots"
    
    # Verbosity
    verbose: bool = True
    debug_mode: bool = False


# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

class ExperimentConfig:
    """Experiment tracking and reproducibility."""
    
    # Random seeds
    random_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42
    
    # Experiment metadata
    experiment_name: str = "FCL_V4_Publication"
    experiment_version: str = "1.0"
    
    # Reproducibility
    deterministic: bool = True
    benchmark_cudnn: bool = False


# ============================================================================
# COMPOSITE CONFIGURATION CLASS
# ============================================================================

class FCLConfig:
    """Master configuration object combining all sub-configs."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.multimodal = MultimodalConfig()
        self.der = DERConfig()
        self.training = TrainingConfig()
        self.continual = ContinualLearningConfig()
        self.federated = FederatedConfig()
        self.privacy = PrivacyConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        self.experiment = ExperimentConfig()
    
    def to_dict(self) -> Dict:
        """Convert all configs to dictionary."""
        return {
            "model": self.model.__dict__,
            "multimodal": self.multimodal.__dict__,
            "der": self.der.__dict__,
            "training": self.training.__dict__,
            "continual": self.continual.__dict__,
            "federated": self.federated.__dict__,
            "privacy": self.privacy.__dict__,
            "data": self.data.__dict__,
            "logging": self.logging.__dict__,
            "experiment": self.experiment.__dict__,
        }
    
    def __repr__(self) -> str:
        """Pretty-print configuration."""
        lines = ["=" * 70, "FEDERATED CONTINUAL LEARNING CONFIGURATION", "=" * 70]
        for section_name, section_config in self.to_dict().items():
            lines.append(f"\n[{section_name.upper()}]")
            for key, value in section_config.items():
                if not key.startswith('_'):
                    lines.append(f"  {key}: {value}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Global configuration instance (used throughout the project)
config = FCLConfig()

# Backward compatibility alias
DEFAULT_CONFIG = config

if __name__ == "__main__":
    # Print full configuration when script is run directly
    print(config)
