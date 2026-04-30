import torch
import torch.nn as nn
from code.model import FTTransformer
from code.multimodal.image_extractor import MobileNetExtractor
from code.config import FCLConfig, ModelConfig, MultimodalConfig, TrainingConfig


class MultimodalFCLModel(nn.Module):
    """
    Multimodal Federated Continual Learning Model.
    Fuses features from clinical images (CNN) and EHR tabular data (Transformer).

    Supports two calling conventions:
      - MultimodalFCLModel(config: FCLConfig)
      - MultimodalFCLModel(model_config=..., multimodal_config=...)
    """

    def __init__(self, config=None, multimodal_config=None, *, model_config=None):
        super().__init__()

        if model_config is not None:
            mc = model_config
            mmc = multimodal_config if multimodal_config is not None else MultimodalConfig()
            tc = TrainingConfig()
        elif isinstance(config, FCLConfig):
            mc = config.model
            mmc = config.multimodal
            tc = config.training
        elif isinstance(config, ModelConfig):
            mc = config
            mmc = multimodal_config if multimodal_config is not None else MultimodalConfig()
            tc = TrainingConfig()
        else:
            mc = ModelConfig()
            mmc = MultimodalConfig()
            tc = TrainingConfig()

        self.model_config = mc
        self.multimodal_config = mmc

        backbone_map = {
            'mobilenet_small': 'mobilenet_v3_small',
            'mobilenet_large': 'mobilenet_v3_large',
            'mobilenet_v3_small': 'mobilenet_v3_small',
            'mobilenet_v3_large': 'mobilenet_v3_large',
        }
        backbone_type = backbone_map.get(mmc.cnn_backbone, 'mobilenet_v3_small')

        self.image_branch = MobileNetExtractor(
            backbone_type=backbone_type,
            pretrained=False
        )

        self.tabular_branch = FTTransformer(mc, tc)

        img_dim = self.image_branch.feature_dim
        tab_dim = mc.token_dim
        combined_dim = img_dim + tab_dim
        hidden_dim = mmc.hidden_dim
        output_dim = mc.output_dim
        dropout = getattr(mmc, 'dropout', mc.mlp_dropout)

        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion_mlp = nn.Sequential(
            self.fusion_layer,
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.tabular_head = nn.Linear(tab_dim, output_dim)

    def forward(self, tabular_or_images: torch.Tensor, images_or_tabular: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass. Accepts either:
          - (tabular,) — tabular only
          - (tabular, images) — tabular first, images second
          - (images, tabular) — images first, tabular second (auto-detected by dim)
        """
        if images_or_tabular is None:
            tab_features = self.tabular_branch.extract_features(tabular_or_images)
            return self.tabular_head(tab_features)

        a, b = tabular_or_images, images_or_tabular
        if a.dim() == 4 and b.dim() == 2:
            images, tabular = a, b
        elif a.dim() == 2 and b.dim() == 4:
            tabular, images = a, b
        else:
            tabular, images = a, b

        img_features = self.image_branch(images)
        tab_features = self.tabular_branch.extract_features(tabular)
        combined = torch.cat([img_features, tab_features], dim=1)
        logits = self.fusion_mlp(combined)
        return logits
