import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict

class MobileNetExtractor(nn.Module):
    """
    MobileNetV3-based image feature extractor for clinical images.
    
    Supports multiple MobileNetV3 backbone variants with automatic feature
    dimension detection. Outputs fixed-size feature vectors suitable for
    multimodal fusion with tabular data.
    
    Supported Backbones:
    - mobilenet_v3_small: 576 features (default, lightweight)
    - mobilenet_v3_large: 960 features (more expressive, higher compute)
    
    Architecture:
    1. Load pretrained MobileNetV3 backbone
    2. Remove classification head (keep feature extraction layers)
    3. Global Average Pooling → feature vector
    4. Output: (batch_size, feature_dim)
    
    For clinical/medical imaging:
    - Uses ImageNet pretraining as initialization (transfer learning)
    - Fine-tuned on downstream medical image datasets (CheXpert, etc.)
    - Efficient for real-time inference and federated scenarios
    """
    
    def __init__(self, backbone_type: str = "mobilenet_v3_small", pretrained: bool = True):
        """
        Initialize MobileNetV3 feature extractor.
        
        Args:
            backbone_type: Type of MobileNetV3 backbone
                - "mobilenet_v3_small": 576-dim features, ~2.5M parameters
                - "mobilenet_v3_large": 960-dim features, ~5.4M parameters
            pretrained: If True, load weights from ImageNet pretraining
        
        Raises:
            ValueError: If backbone_type not supported
        
        Examples:
            >>> extractor = MobileNetExtractor("mobilenet_v3_small", pretrained=True)
            >>> print(extractor.feature_dim)  # 576
            >>> x = torch.randn(32, 3, 224, 224)
            >>> features = extractor(x)
            >>> features.shape
            torch.Size([32, 576])
        """
        super().__init__()
        
        # Load backbone with automatic feature dimension detection
        if backbone_type == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        elif backbone_type == "mobilenet_v3_large":
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_type}. "
                f"Supported: mobilenet_v3_small, mobilenet_v3_large"
            )
        
        self.backbone_type = backbone_type
        
        # Dynamically detect feature dimension
        # Extract from classifier input dimension before replacing it
        self.feature_dim = self._get_feature_dim()
        
        # Remove the classification head, keep as Identity
        # MobileNetV3 has: features → avgpool → classifier
        # We want: features → avgpool → [our output]
        self.backbone.classifier = nn.Identity()
    
    def _get_feature_dim(self) -> int:
        """
        Automatically detect feature dimension for current backbone.
        
        Uses a dummy forward pass to infer the output size without relying
        on hardcoded values. This approach is robust to different model
        versions and configurations.
        
        Returns:
            int: Number of output features from backbone
        
        Implementation:
        1. Create dummy input tensor (1, 3, 224, 224)
        2. Run through backbone without computing gradients
        3. Extract output dimension from classifier input
        4. Supported variants:
           - MobileNetV3-Small: 576
           - MobileNetV3-Large: 960
           - Custom/pruned variants: auto-detected
        
        Examples:
            >>> extractor = MobileNetExtractor()
            >>> dim = extractor._get_feature_dim()
            >>> print(f"Feature dimension: {dim}")
            Feature dimension: 576
        """
        # Get the classifier's input features (before we replaced it)
        try:
            # For MobileNetV3, classifier is a Sequential module
            # classifier[0] is Linear layer that takes features
            classifier_in_features = self.backbone.classifier[0].in_features
            return classifier_in_features
        except (IndexError, AttributeError) as e:
            # Fallback: use dummy forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                # Get features before classifier
                features = self.backbone.features(dummy_input)
                features = self.backbone.avgpool(features)
                feature_dim = features.view(1, -1).shape[1]
            return feature_dim
    
    @property
    def backbone_variants(self) -> Dict[str, int]:
        """
        Reference dictionary of known backbone variants and their feature dims.
        
        Returns:
            Dict mapping backbone name to feature dimension
        """
        return {
            "mobilenet_v3_small": 576,
            "mobilenet_v3_large": 960,
            "mobilenet_v3_small_320": 576,  # Reduced input size variant
            "mobilenet_v3_large_320": 960,
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from clinical images.
        
        Input images are typically from:
        - Medical imaging modalities: X-ray, CT, MRI, ultrasound
        - Preprocessed to 224×224 resolution (ImageNet standard)
        - Normalized with ImageNet statistics
        
        Args:
            x: (batch_size, 3, 224, 224) - RGB medical images
               Values typically in [0, 1] or [-1, 1] depending on normalization
        
        Returns:
            (batch_size, feature_dim) - fixed-size feature vectors
                Where feature_dim = 576 (small) or 960 (large)
        
        Examples:
            >>> extractor = MobileNetExtractor("mobilenet_v3_small")
            >>> images = torch.randn(32, 3, 224, 224)
            >>> features = extractor(images)
            >>> features.shape
            torch.Size([32, 576])
            
            >>> # Use in multimodal fusion
            >>> img_features = extractor(images)  # (batch, 576)
            >>> combined = torch.cat([img_features, tab_features], dim=1)
        """
        # Forward through backbone
        # MobileNetV3: features → avgpool → classifier → output
        # Since classifier = Identity, we get the avgpool output
        features = self.backbone(x)
        
        # The output from Identity classifier preserves the avgpool shape
        # avgpool reduces spatial dimensions: (batch, channels, 1, 1)
        # Flatten to vector: (batch, channels)
        features_flat = features.view(features.size(0), -1)
        
        return features_flat
