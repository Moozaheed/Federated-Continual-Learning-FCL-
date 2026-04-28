"""
FT-Transformer Model Implementation
Feature Tokenizer Transformer for tabular clinical data.
Designed for Federated Continual Learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List
from code.config import ModelConfig, TrainingConfig


class FeatureTokenizer(nn.Module):
    """
    Converts numerical features into learnable token embeddings.
    Each feature gets its own embedding vector.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_dim = config.input_dim
        self.token_dim = config.token_dim
        
        # Feature embeddings: one embedding per feature
        self.embeddings = nn.Linear(1, config.token_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) - raw feature values
        
        Returns:
            (batch_size, input_dim, token_dim) - feature tokens
        """
        batch_size = x.shape[0]
        
        # Reshape for embedding: (batch_size * input_dim, 1)
        x_reshaped = x.reshape(-1, 1)
        
        # Apply embedding
        tokens = self.embeddings(x_reshaped)  # (batch_size * input_dim, token_dim)
        
        # Reshape back to (batch_size, input_dim, token_dim)
        tokens = tokens.reshape(batch_size, self.input_dim, self.token_dim)
        
        return tokens


class PromptTuningModule(nn.Module):
    """
    Learnable prompt tokens for task-specific adaptation.
    Enables parameter-efficient continual learning.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_prompt_tokens = config.n_prompt_tokens
        self.token_dim = config.token_dim
        
        # Learnable prompt tokens
        self.prompts = nn.Parameter(
            torch.randn(1, config.n_prompt_tokens, config.token_dim) * config.prompt_init_std
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Concatenate prompt tokens to feature tokens.
        
        Args:
            x: (batch_size, n_features, token_dim)
        
        Returns:
            (batch_size, n_features + n_prompt_tokens, token_dim)
        """
        batch_size = x.shape[0]
        
        # Expand prompts to batch size
        prompts = self.prompts.expand(batch_size, -1, -1)
        
        # Concatenate along sequence dimension
        return torch.cat([x, prompts], dim=1)


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention mechanism."""
    
    def __init__(self, token_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert token_dim % n_heads == 0, "token_dim must be divisible by n_heads"
        
        self.token_dim = token_dim
        self.n_heads = n_heads
        self.head_dim = token_dim // n_heads
        
        self.qkv = nn.Linear(token_dim, 3 * token_dim)
        self.proj = nn.Linear(token_dim, token_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, token_dim)
        
        Returns:
            (batch_size, seq_len, token_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V projections
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = (attn @ v).transpose(1, 2)  # (batch_size, seq_len, n_heads, head_dim)
        context = context.reshape(batch_size, seq_len, self.token_dim)
        
        # Output projection
        output = self.proj(context)
        output = self.dropout(output)
        
        return output


class TransformerBlock(nn.Module):
    """Single Transformer block: Attention + MLP + LayerNorm."""
    
    def __init__(self, token_dim: int, n_heads: int, mlp_hidden_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Attention
        self.norm1 = nn.LayerNorm(token_dim)
        self.attention = MultiHeadAttention(token_dim, n_heads, dropout)
        
        # MLP
        self.norm2 = nn.LayerNorm(token_dim)
        mlp_hidden = token_dim * mlp_hidden_factor
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, token_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP with residual connections."""
        # Attention block with residual
        x = x + self.attention(self.norm1(x))
        
        # MLP block with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for clinical tabular data.
    
    Architecture:
    1. Feature Tokenization: Convert each feature to a token
    2. Prompt Tuning: Add learnable task-specific tokens
    3. Transformer Blocks: Multi-head attention + MLP
    4. Classification Head: Linear layer for output
    """
    
    def __init__(self, config: ModelConfig, training_config: TrainingConfig):
        super().__init__()
        self.config = config
        self.training_config = training_config
        
        # Components
        self.feature_tokenizer = FeatureTokenizer(config)
        self.prompt_tuning = PromptTuningModule(config)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                token_dim=config.token_dim,
                n_heads=config.n_attention_heads,
                mlp_hidden_factor=config.mlp_hidden_factor,
                dropout=config.mlp_dropout
            )
            for _ in range(config.n_transformer_blocks)
        ])
        
        # Final layers
        self.final_norm = nn.LayerNorm(config.token_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(config.token_dim, config.token_dim // 2),
            nn.GELU(),
            nn.Dropout(config.mlp_dropout),
            nn.Linear(config.token_dim // 2, config.output_dim)
        )
        
        # Fisher information matrix (for EWC)
        self.register_buffer('fisher_matrix', None)
        self.register_buffer('task_specific_weights', None)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FT-Transformer.
        
        Args:
            x: (batch_size, input_dim) - clinical features
        
        Returns:
            logits: (batch_size, output_dim) - class logits
        """
        # Feature tokenization
        tokens = self.feature_tokenizer(x)  # (batch_size, input_dim, token_dim)
        
        # Prompt tuning
        tokens = self.prompt_tuning(tokens)  # (batch_size, input_dim + n_prompt, token_dim)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)
        
        # Use first token ([CLS] token semantics) for classification
        cls_token = tokens[:, 0, :]  # (batch_size, token_dim)
        
        # Final normalization
        cls_token = self.final_norm(cls_token)
        
        # Classification head
        logits = self.head(cls_token)  # (batch_size, output_dim)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract attention maps from each block for interpretability.
        
        Returns:
            Dict mapping block_idx -> attention_map
        """
        attention_maps = {}
        # Implementation would hook into attention modules
        return attention_maps
    
    def compute_fisher_information(self, data_loader, loss_fn, device: str = "cpu"):
        """
        Compute Fisher Information Matrix for Elastic Weight Consolidation (EWC).
        
        Args:
            data_loader: DataLoader with training data
            loss_fn: Loss function
            device: Device to compute on
        """
        fisher_matrix = {}
        
        for name, param in self.named_parameters():
            fisher_matrix[name] = torch.zeros_like(param.data)
        
        self.train()
        
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            logits = self(batch_x)
            loss = loss_fn(logits, batch_y)
            
            # Backward pass
            self.zero_grad()
            loss.backward(retain_graph=True)
            
            # Accumulate squared gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_matrix[name] += param.grad ** 2
        
        # Average over samples
        for name in fisher_matrix:
            fisher_matrix[name] /= len(data_loader.dataset)
        
        self.fisher_matrix = fisher_matrix
        return fisher_matrix
    
    def get_prompt_tokens(self) -> torch.Tensor:
        """Extract prompt tokens (for parameter efficiency analysis)."""
        return self.prompt_tuning.prompts.clone()
    
    def get_param_count(self) -> int:
        """Get total parameter count (backward compatibility alias)."""
        return self.total_parameters
    
    @property
    def total_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def prompt_parameters(self) -> int:
        """Count prompt-tuning parameters only."""
        return self.prompt_tuning.prompts.numel()


def create_model(
    config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    device: str = "cpu",
    # Backward compatibility keyword arguments
    num_numerical_features: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    num_transformer_blocks: Optional[int] = None,
    num_classes: Optional[int] = None,
    num_prompts: Optional[int] = None,
    use_prompts: Optional[bool] = None,
    **kwargs
) -> FTTransformer:
    """
    Factory function to create FT-Transformer model.
    
    Supports both:
    1. Config object approach: create_model(config, training_config, device)
    2. Keyword argument approach: create_model(num_numerical_features=13, num_classes=2, ...)
    
    Args:
        config: Model configuration object (or None to use kwargs)
        training_config: Training configuration object
        device: Device to create model on
        num_numerical_features: Input dimension (backward compat)
        embedding_dim: Token/embedding dimension (backward compat)
        num_transformer_blocks: Number of transformer blocks (backward compat)
        num_classes: Output dimension (backward compat)
        num_prompts: Number of prompt tokens (backward compat)
        use_prompts: Whether to use prompts (backward compat)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        FTTransformer instance
    """
    # If config not provided, use keyword arguments
    if config is None:
        config = ModelConfig()
        if num_numerical_features is not None:
            config.input_dim = num_numerical_features
        if embedding_dim is not None:
            config.token_dim = embedding_dim
        if num_transformer_blocks is not None:
            config.n_transformer_blocks = num_transformer_blocks
        if num_classes is not None:
            config.output_dim = num_classes
        if num_prompts is not None:
            config.n_prompt_tokens = num_prompts
        if use_prompts is False:
            config.n_prompt_tokens = 0
    
    # If training_config not provided, use default
    if training_config is None:
        training_config = TrainingConfig()
    
    model = FTTransformer(config, training_config)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Quick test
    from code.config import ModelConfig, TrainingConfig
    
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    model = create_model(model_config, train_config, device="cpu")
    
    print(f"Model created successfully!")
    print(f"Total parameters: {model.total_parameters:,}")
    print(f"Prompt parameters: {model.prompt_parameters:,}")
    
    # Test forward pass
    x = torch.randn(32, 13)  # Batch of 32 samples, 13 features
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (32, 2)
