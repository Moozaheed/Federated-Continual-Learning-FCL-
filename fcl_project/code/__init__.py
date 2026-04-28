"""
Federated Continual Learning (FCL) Framework
Code package for enterprise-grade FCL research implementation.
"""

__version__ = "1.0.0"
__author__ = "FCL Research Team"

from code.config import (
    FCLConfig,
    ModelConfig,
    TrainingConfig,
    ContinualLearningConfig,
    FederatedConfig,
    PrivacyConfig,
    DataConfig,
    config,
)

from code.model import (
    FTTransformer,
    FeatureTokenizer,
    PromptTuningModule,
    MultiHeadAttention,
    TransformerBlock,
    create_model,
)

from code.utils import (
    load_uci_heart_disease,
    create_non_iid_splits,
    create_data_loaders,
    compute_metrics,
    compute_backward_transfer,
    compute_forward_transfer,
    plot_training_history,
    plot_confusion_matrix,
    aggregate_gradients,
    extract_model_gradients,
    compute_epsilon_from_noise,
)

__all__ = [
    # Config
    'FCLConfig',
    'ModelConfig',
    'TrainingConfig',
    'ContinualLearningConfig',
    'FederatedConfig',
    'PrivacyConfig',
    'DataConfig',
    'config',
    # Model
    'FTTransformer',
    'FeatureTokenizer',
    'PromptTuningModule',
    'MultiHeadAttention',
    'TransformerBlock',
    'create_model',
    # Utils
    'load_uci_heart_disease',
    'create_non_iid_splits',
    'create_data_loaders',
    'compute_metrics',
    'compute_backward_transfer',
    'compute_forward_transfer',
    'plot_training_history',
    'plot_confusion_matrix',
    'aggregate_gradients',
    'extract_model_gradients',
    'compute_epsilon_from_noise',
]
