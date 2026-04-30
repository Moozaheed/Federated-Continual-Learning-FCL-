import torch
import torch.nn as nn
import torchvision.models as models
import os
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
    
    def __init__(
        self,
        backbone_type: str = "mobilenet_v3_small",
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        model_type: Optional[str] = None
    ):
        """
        Initialize MobileNetV3 feature extractor.
        
        Args:
            backbone_type: Type of MobileNetV3 backbone
                - "mobilenet_v3_small": 576-dim features, ~2.5M parameters
                - "mobilenet_v3_large": 960-dim features, ~5.4M parameters
            pretrained: If True, load weights from ImageNet pretraining
            weights_path: Optional path to local weights file for offline environments.
                If provided, will load from this path instead of downloading.
                Useful for hospital systems without internet access.
        
        Raises:
            ValueError: If backbone_type not supported
            FileNotFoundError: If weights_path provided but file not found
        
        Examples:
            >>> # Online mode
            >>> extractor = MobileNetExtractor("mobilenet_v3_small", pretrained=True)
            
            >>> # Offline mode with local weights
            >>> extractor = MobileNetExtractor(
            ...     "mobilenet_v3_small",
            ...     pretrained=False,
            ...     weights_path="/hospital/models/mobilenet_v3_small.pth"
            ... )
        """
        super().__init__()

        if model_type is not None:
            type_map = {'small': 'mobilenet_v3_small', 'large': 'mobilenet_v3_large'}
            backbone_type = type_map.get(model_type, f'mobilenet_v3_{model_type}')

        self.backbone_type = backbone_type
        backbone_fn = {
            "mobilenet_v3_small": models.mobilenet_v3_small,
            "mobilenet_v3_large": models.mobilenet_v3_large,
        }.get(backbone_type)
        if backbone_fn is None:
            raise ValueError(
                f"Unsupported backbone: {backbone_type}. "
                f"Supported: mobilenet_v3_small, mobilenet_v3_large"
            )

        if weights_path is not None:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Weights file not found at {weights_path}. "
                    f"Please ensure the file exists for offline operation."
                )
            self.backbone = backbone_fn(pretrained=False)
            self.backbone.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)
        else:
            self.backbone = backbone_fn(pretrained=pretrained)
        
        # Dynamically detect feature dimension
        self.feature_dim = self._get_feature_dim()

        # Remove the classification head
        self.backbone.classifier = nn.Identity()

        self.model = self.backbone
    
    @staticmethod
    def download_pretrained_weights(backbone_type: str = "mobilenet_v3_small", save_path: str = "/hospital/models/") -> str:
        """
        Download and cache pretrained weights for offline use.
        
        Useful for hospital systems to pre-download weights before deployment
        in offline environments.
        
        Args:
            backbone_type: Which backbone to download
            save_path: Directory to save weights
        
        Returns:
            Path to saved weights file
        
        Examples:
            >>> path = MobileNetExtractor.download_pretrained_weights(
            ...     backbone_type="mobilenet_v3_small",
            ...     save_path="/hospital/models/"
            ... )
            >>> print(f"Weights saved to {path}")
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Load model (triggers download if not cached)
        if backbone_type == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(pretrained=True)
        elif backbone_type == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")
        
        # Save to disk
        weights_file = os.path.join(save_path, f"{backbone_type}_pretrained.pth")
        torch.save(model.state_dict(), weights_file)
        
        print(f"✓ Weights downloaded and saved to {weights_file}")
        return weights_file
    
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
