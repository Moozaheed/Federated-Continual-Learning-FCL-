import torch
import torch.nn as nn
from code.model import FTTransformer
from code.multimodal.image_extractor import MobileNetExtractor
from code.config import FCLConfig

class MultimodalFCLModel(nn.Module):
    """
    Multimodal Federated Continual Learning Model.
    Fuses features from clinical images (CNN) and EHR tabular data (Transformer).
    """
    def __init__(self, config: FCLConfig):
        super().__init__()
        self.config = config
        
        # Branch A: Image Extractor
        self.image_branch = MobileNetExtractor(
            backbone_type=config.multimodal.cnn_backbone,
            pretrained=True
        )
        
        # Branch B: Tabular Extractor (FT-Transformer)
        self.tabular_branch = FTTransformer(config.model, config.training)
        # Modify tabular branch to return features instead of logits
        # We'll use the internal components of FTTransformer
        
        # Fusion Layer
        combined_dim = self.image_branch.feature_dim + config.model.token_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, config.multimodal.hidden_dim),
            nn.BatchNorm1d(config.multimodal.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model.mlp_dropout),
            nn.Linear(config.multimodal.hidden_dim, config.multimodal.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.multimodal.hidden_dim // 2, config.model.output_dim)
        )
        
    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, 3, 224, 224)
            tabular: (batch_size, input_dim)
        Returns:
            logits: (batch_size, output_dim)
        """
        # Extract image features: (batch_size, 576)
        img_features = self.image_branch(images)
        
        # Extract tabular features from FT-Transformer (using its CLS token logic)
        # We need to reach into FTTransformer's forward logic or modify it.
        # For now, let's reuse the logic:
        tokens = self.tabular_branch.feature_tokenizer(tabular)
        tokens = self.tabular_branch.prompt_tuning(tokens)
        for block in self.tabular_branch.transformer_blocks:
            tokens = block(tokens)
        tab_features = self.tabular_branch.final_norm(tokens[:, 0, :]) # (batch_size, token_dim)
        
        # Concat fusion
        combined = torch.cat([img_features, tab_features], dim=1)
        
        # Final prediction
        logits = self.fusion_mlp(combined)
        return logits
