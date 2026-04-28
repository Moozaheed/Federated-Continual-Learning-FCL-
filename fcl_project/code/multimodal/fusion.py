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
        Forward pass through multimodal fusion.
        
        Architecture:
        1. Image Branch: MobileNetV3 CNN → feature_dim features (e.g., 576)
        2. Tabular Branch: FT-Transformer → token_dim features (e.g., 128)
        3. Fusion: Concatenate features → MLP → logits
        
        Args:
            images: (batch_size, 3, 224, 224) - RGB clinical images
            tabular: (batch_size, input_dim) - EHR tabular data (13 clinical features)
        
        Returns:
            logits: (batch_size, output_dim) - classification logits
        
        Examples:
            >>> model = MultimodalFCLModel(config)
            >>> images = torch.randn(32, 3, 224, 224)
            >>> tabular = torch.randn(32, 13)
            >>> logits = model(images, tabular)
            >>> logits.shape
            torch.Size([32, 2])
        """
        # Extract image features: (batch_size, feature_dim)
        # MobileNetV3-Small default: feature_dim=576
        img_features = self.image_branch(images)
        
        # Extract tabular features using the new extract_features() method
        # This returns (batch_size, token_dim) without passing through classification head
        tab_features = self.tabular_branch.extract_features(tabular)
        
        # Concatenate both modalities: (batch_size, feature_dim + token_dim)
        combined = torch.cat([img_features, tab_features], dim=1)
        
        # Apply fusion MLP: (batch_size, feature_dim + token_dim) → (batch_size, output_dim)
        logits = self.fusion_mlp(combined)
        
        return logits
