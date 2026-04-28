import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class MobileNetExtractor(nn.Module):
    """
    MobileNetV3-based image feature extractor for clinical images.
    """
    def __init__(self, backbone_type: str = "mobilenet_v3_small", pretrained: bool = True):
        super().__init__()
        
        if backbone_type == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        elif backbone_type == "mobilenet_v3_large":
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")
            
        # Remove the classification head
        # MobileNetV3 has a 'classifier' attribute
        self.feature_dim = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 3, H, W) image tensor
        Returns:
            (batch_size, feature_dim) feature vector
        """
        # MobileNetV3 output after Identity classifier will be (batch_size, feature_dim, 1, 1) 
        # due to Global Average Pooling before the classifier
        features = self.backbone(x)
        return features.view(features.size(0), -1)
