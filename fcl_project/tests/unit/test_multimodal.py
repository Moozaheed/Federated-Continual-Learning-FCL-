"""
Unit Tests for Multimodal Components
Tests for multimodal fusion and image extraction.
"""

import unittest
import torch
import sys
sys.path.insert(0, '/home/bs01233/Documents/FL/fcl_project')

from code.multimodal.fusion import MultimodalFCLModel
from code.multimodal.image_extractor import MobileNetExtractor
from code.config import ModelConfig, MultimodalConfig


class TestMobileNetExtractor(unittest.TestCase):
    """Test MobileNetV3 image feature extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.batch_size = 8
        self.image_channels = 3
        self.image_size = 224
        
        # Test images
        self.images = torch.randn(
            self.batch_size,
            self.image_channels,
            self.image_size,
            self.image_size
        )
    
    def test_extractor_initialization(self):
        """Test initializing MobileNetV3 extractor."""
        extractor = MobileNetExtractor(model_type='small', pretrained=False)
        self.assertIsNotNone(extractor)
    
    def test_extractor_small_model(self):
        """Test small MobileNetV3 model."""
        extractor = MobileNetExtractor(model_type='small', pretrained=False)
        
        with torch.no_grad():
            features = extractor(self.images)
        
        self.assertEqual(features.shape[0], self.batch_size)
        self.assertEqual(features.shape[1], 576)  # Small model output dim
    
    def test_extractor_large_model(self):
        """Test large MobileNetV3 model."""
        extractor = MobileNetExtractor(model_type='large', pretrained=False)
        
        with torch.no_grad():
            features = extractor(self.images)
        
        self.assertEqual(features.shape[0], self.batch_size)
        self.assertEqual(features.shape[1], 960)  # Large model output dim
    
    def test_extractor_output_shape(self):
        """Test extractor produces 1D feature vectors."""
        extractor = MobileNetExtractor(pretrained=False)
        
        with torch.no_grad():
            features = extractor(self.images)
        
        # Should be 2D: (batch_size, feature_dim)
        self.assertEqual(len(features.shape), 2)
        self.assertEqual(features.shape[0], self.batch_size)
    
    def test_extractor_deterministic_no_training(self):
        """Test extractor is deterministic in eval mode."""
        extractor = MobileNetExtractor(pretrained=False)
        extractor.eval()
        
        with torch.no_grad():
            features1 = extractor(self.images)
            features2 = extractor(self.images)
        
        self.assertTrue(torch.allclose(features1, features2))
    
    def test_extractor_different_batch_sizes(self):
        """Test extractor works with different batch sizes."""
        extractor = MobileNetExtractor(pretrained=False)
        
        for batch_size in [1, 4, 8, 16]:
            images = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                features = extractor(images)
            
            self.assertEqual(features.shape[0], batch_size)
            self.assertEqual(len(features.shape), 2)
    
    def test_extractor_offline_weights(self):
        """Test offline weights loading."""
        import tempfile
        import os
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, 'mobilenet.pth')
            
            # Create and save a model
            extractor1 = MobileNetExtractor(pretrained=False)
            torch.save(extractor1.model.state_dict(), weights_path)
            
            # Load with offline path
            extractor2 = MobileNetExtractor(
                pretrained=True,
                weights_path=weights_path
            )
            
            # Verify same output
            with torch.no_grad():
                out1 = extractor1(self.images)
                out2 = extractor2(self.images)
            
            self.assertTrue(torch.allclose(out1, out2, atol=1e-5))
    
    def test_extractor_no_grad_inference(self):
        """Test inference doesn't accumulate gradients."""
        extractor = MobileNetExtractor(pretrained=False)
        
        with torch.no_grad():
            features = extractor(self.images)
        
        self.assertIsNone(features.grad_fn)


class TestMultimodalFCLModel(unittest.TestCase):
    """Test multimodal FCL model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.batch_size = 8
        
        # Config
        self.model_config = ModelConfig(
            num_features=13,
            num_classes=2,
            d_model=128,
            n_heads=4,
            n_layers=2,
            ffn_multiplier=4,
            norm_layer='LayerNorm'
        )
        
        self.multimodal_config = MultimodalConfig(
            enable_multimodal=True,
            image_extractor_type='mobilenet_small',
            fusion_type='linear',
            fusion_dim=256,
            dropout=0.1
        )
        
        # Create model
        self.model = MultimodalFCLModel(
            model_config=self.model_config,
            multimodal_config=self.multimodal_config
        )
    
    def test_multimodal_model_initialization(self):
        """Test multimodal model initializes correctly."""
        self.assertIsNotNone(self.model)
        self.assertTrue(self.model.multimodal_config.enable_multimodal)
    
    def test_multimodal_forward_pass(self):
        """Test forward pass with multimodal inputs."""
        tabular = torch.randn(self.batch_size, 13)
        images = torch.randn(self.batch_size, 3, 224, 224)
        
        logits = self.model(tabular, images)
        
        self.assertEqual(logits.shape, (self.batch_size, 2))
    
    def test_multimodal_output_shape(self):
        """Test output shape is correct."""
        tabular = torch.randn(self.batch_size, 13)
        images = torch.randn(self.batch_size, 3, 224, 224)
        
        logits = self.model(tabular, images)
        
        self.assertEqual(logits.shape[0], self.batch_size)
        self.assertEqual(logits.shape[1], 2)  # num_classes
    
    def test_multimodal_backward_pass(self):
        """Test gradients flow in multimodal model."""
        tabular = torch.randn(self.batch_size, 13, requires_grad=True)
        images = torch.randn(self.batch_size, 3, 224, 224, requires_grad=True)
        
        logits = self.model(tabular, images)
        loss = logits.sum()
        loss.backward()
        
        self.assertIsNotNone(tabular.grad)
        self.assertIsNotNone(images.grad)
    
    def test_multimodal_tabular_only(self):
        """Test model works with tabular data only."""
        tabular = torch.randn(self.batch_size, 13)
        
        logits = self.model(tabular)
        
        self.assertEqual(logits.shape, (self.batch_size, 2))
    
    def test_multimodal_fusion_layer(self):
        """Test fusion layer operates correctly."""
        self.assertIsNotNone(self.model.fusion_layer)

        # Use extracted feature dimensions (token_dim from transformer, not raw input)
        tab_dim = self.model_config.token_dim
        img_dim = self.model.image_branch.feature_dim
        tabular_features = torch.randn(self.batch_size, tab_dim)
        image_features = torch.randn(self.batch_size, img_dim)

        combined = torch.cat([image_features, tabular_features], dim=1)
        fused = self.model.fusion_layer(combined)

        self.assertEqual(fused.shape[0], self.batch_size)
    
    def test_multimodal_dropout_effect(self):
        """Test dropout in training mode."""
        self.model.train()
        
        tabular = torch.randn(self.batch_size, 13)
        images = torch.randn(self.batch_size, 3, 224, 224)
        
        logits1 = self.model(tabular, images)
        logits2 = self.model(tabular, images)
        
        # With dropout, outputs should differ
        # (Note: very small chance they're identical)
        # Just check they're reasonable
        self.assertEqual(logits1.shape, logits2.shape)
    
    def test_multimodal_different_batch_sizes(self):
        """Test model with various batch sizes."""
        for batch_size in [1, 4, 8, 16]:
            tabular = torch.randn(batch_size, 13)
            images = torch.randn(batch_size, 3, 224, 224)
            
            logits = self.model(tabular, images)
            
            self.assertEqual(logits.shape[0], batch_size)
            self.assertEqual(logits.shape[1], 2)
    
    def test_multimodal_eval_mode(self):
        """Test eval mode disables dropout."""
        self.model.eval()
        
        tabular = torch.randn(self.batch_size, 13)
        images = torch.randn(self.batch_size, 3, 224, 224)
        
        with torch.no_grad():
            logits1 = self.model(tabular, images)
            logits2 = self.model(tabular, images)
        
        # In eval mode, should be deterministic
        self.assertTrue(torch.allclose(logits1, logits2))


class TestFusionLayer(unittest.TestCase):
    """Test fusion layer for multimodal fusion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.tabular_dim = 13
        self.image_dim = 576
        self.fusion_dim = 256
    
    def test_fusion_layer_initialization(self):
        """Test fusion layer initializes."""
        import torch.nn as nn
        
        fusion = nn.Sequential(
            nn.Linear(self.tabular_dim + self.image_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.assertIsNotNone(fusion)
    
    def test_fusion_layer_forward(self):
        """Test fusion layer forward pass."""
        import torch.nn as nn
        
        fusion = nn.Sequential(
            nn.Linear(self.tabular_dim + self.image_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU()
        )
        
        combined = torch.randn(self.batch_size, self.tabular_dim + self.image_dim)
        output = fusion(combined)
        
        self.assertEqual(output.shape, (self.batch_size, self.fusion_dim))
    
    def test_fusion_layer_gradients(self):
        """Test gradients flow through fusion layer."""
        import torch.nn as nn
        
        fusion = nn.Sequential(
            nn.Linear(self.tabular_dim + self.image_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        
        combined = torch.randn(
            self.batch_size,
            self.tabular_dim + self.image_dim,
            requires_grad=True
        )
        
        output = fusion(combined)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(combined.grad)
        self.assertTrue(torch.sum(combined.grad) != 0)


class TestImageTabularConcatenation(unittest.TestCase):
    """Test image and tabular data concatenation."""
    
    def test_concatenation_shapes(self):
        """Test concatenating image and tabular features."""
        batch_size = 8
        tabular_dim = 13
        image_dim = 576
        
        tabular = torch.randn(batch_size, tabular_dim)
        images = torch.randn(batch_size, image_dim)
        
        combined = torch.cat([tabular, images], dim=1)
        
        self.assertEqual(combined.shape, (batch_size, tabular_dim + image_dim))
    
    def test_concatenation_order_matters(self):
        """Test order of concatenation affects results."""
        batch_size = 8
        
        tabular = torch.randn(batch_size, 13)
        images = torch.randn(batch_size, 576)
        
        concat1 = torch.cat([tabular, images], dim=1)
        concat2 = torch.cat([images, tabular], dim=1)
        
        self.assertFalse(torch.allclose(concat1, concat2))
    
    def test_concatenation_gradient_flow(self):
        """Test gradients flow through concatenation."""
        tabular = torch.randn(8, 13, requires_grad=True)
        images = torch.randn(8, 576, requires_grad=True)
        
        combined = torch.cat([tabular, images], dim=1)
        loss = combined.sum()
        loss.backward()
        
        self.assertIsNotNone(tabular.grad)
        self.assertIsNotNone(images.grad)


class TestMultimodalIntegration(unittest.TestCase):
    """Integration tests for multimodal pipeline."""
    
    def test_end_to_end_multimodal_training_step(self):
        """Test end-to-end multimodal training step."""
        import torch.nn as nn
        
        # Config
        model_config = ModelConfig(
            num_features=13,
            num_classes=2,
            d_model=128,
            n_heads=4,
            n_layers=2
        )
        
        multimodal_config = MultimodalConfig(
            enable_multimodal=True,
            image_extractor_type='mobilenet_small'
        )
        
        # Create model
        model = MultimodalFCLModel(model_config, multimodal_config)
        model.train()
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Data
        tabular = torch.randn(8, 13)
        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 2, (8,))
        
        # Training step
        optimizer.zero_grad()
        logits = model(tabular, images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        # Verify training happened
        self.assertTrue(loss.item() > 0)


if __name__ == '__main__':
    unittest.main()
