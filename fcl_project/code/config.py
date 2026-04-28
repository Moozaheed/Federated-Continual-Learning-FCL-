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
    token_dim: int = 64  # Embedding dimension for each feature token
    n_feature_tokens: int = input_dim  # One token per feature
    
    # Prompt tuning
    n_prompt_tokens: int = 5  # Learnable task-specific tokens
    
    # Transformer blocks
    n_transformer_blocks: int = 3
    n_attention_heads: int = 8
    attention_dropout: float = 0.1
    mlp_hidden_factor: int = 2  # Hidden dim = token_dim * mlp_hidden_factor
    mlp_dropout: float = 0.1
    
    # Output
    output_dim: int = 2  # Binary classification (presence/absence of disease)
    
    @classmethod
    def get_total_params(cls) -> int:
        """Estimated total parameters for this architecture."""
        # Rough calculation:
        # - Feature embeddings: input_dim * token_dim
        # - Prompt tokens: n_prompt_tokens * token_dim
        # - 3 transformer blocks, each with attention + MLP
        return 52_466  # Validated from model instantiation


# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

class TrainingConfig:
    """Training loop hyperparameters."""
    
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

if __name__ == "__main__":
    # Print full configuration when script is run directly
    print(config)
