"""
Unit Tests for Configuration Components
Tests for DERConfig, TrainingConfig, and validation.
"""

import unittest
import sys
sys.path.insert(0, '/home/bs01233/Documents/FL/fcl_project')

from code.config import (
    ModelConfig, TrainingConfig, DERConfig, MultimodalConfig,
    ContinualLearningConfig, FCLConfig
)


class TestDERConfig(unittest.TestCase):
    """Test Dark Experience Replay configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DERConfig()
    
    def test_der_config_defaults(self):
        """Test default DER++ configuration."""
        self.assertEqual(self.config.buffer_size, 5000)
        self.assertEqual(self.config.alpha, 0.3)
        self.assertEqual(self.config.beta, 0.7)
        self.assertTrue(self.config.enabled)
    
    def test_der_config_validate_valid(self):
        """Test validation with valid parameters."""
        config = DERConfig(
            buffer_size=2000,
            alpha=0.4,
            beta=0.6,
            batch_size=32
        )
        
        # Should not raise error
        self.assertTrue(config.validate())
    
    def test_der_config_validate_invalid_alpha(self):
        """Test validation rejects invalid alpha."""
        config = DERConfig(alpha=1.5)  # Out of [0, 1]
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_der_config_validate_invalid_beta(self):
        """Test validation rejects invalid beta."""
        config = DERConfig(beta=-0.1)  # Out of [0, 1]
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_der_config_validate_batch_exceeds_buffer(self):
        """Test validation warns about batch_size > buffer_size."""
        config = DERConfig(buffer_size=100, batch_size=200)
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_der_config_validate_invalid_buffer_size(self):
        """Test validation rejects invalid buffer size."""
        config = DERConfig(buffer_size=0)
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_der_config_get_dict(self):
        """Test getting config as dictionary."""
        config_dict = self.config.get_config_dict()
        
        self.assertIn('enabled', config_dict)
        self.assertIn('buffer_size', config_dict)
        self.assertIn('alpha', config_dict)
        self.assertIn('beta', config_dict)
        self.assertIn('batch_size', config_dict)
    
    def test_der_config_sampling_strategies(self):
        """Test valid sampling strategies."""
        for strategy in ["reservoir", "recent"]:
            config = DERConfig(sampling_strategy=strategy)
            self.assertTrue(config.validate())
    
    def test_der_config_invalid_sampling_strategy(self):
        """Test invalid sampling strategy."""
        config = DERConfig(sampling_strategy="invalid_strategy")
        
        with self.assertRaises(ValueError):
            config.validate()


class TestTrainingConfig(unittest.TestCase):
    """Test training configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TrainingConfig()
    
    def test_training_config_defaults(self):
        """Test default training configuration."""
        self.assertEqual(self.config.learning_rate, 1e-3)
        self.assertEqual(self.config.weight_decay, 1e-4)
        self.assertEqual(self.config.batch_size, 32)
        self.assertEqual(self.config.epochs_per_task, 10)
        self.assertIn(self.config.device, ["cuda", "cpu"])
    
    def test_training_config_modification(self):
        """Test modifying training config."""
        self.config.learning_rate = 5e-4
        self.config.batch_size = 64
        
        self.assertEqual(self.config.learning_rate, 5e-4)
        self.assertEqual(self.config.batch_size, 64)


class TestMultimodalConfig(unittest.TestCase):
    """Test multimodal configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MultimodalConfig()
    
    def test_multimodal_config_defaults(self):
        """Test default multimodal configuration."""
        self.assertTrue(self.config.enabled)
        self.assertEqual(self.config.image_size, (224, 224))
        self.assertEqual(self.config.cnn_backbone, "mobilenet_v3_small")
        self.assertEqual(self.config.fusion_strategy, "concat")
    
    def test_multimodal_config_cnn_backbone_options(self):
        """Test valid CNN backbone options."""
        for backbone in ["mobilenet_v3_small", "mobilenet_v3_large"]:
            config = MultimodalConfig()
            config.cnn_backbone = backbone
            self.assertEqual(config.cnn_backbone, backbone)


class TestContinualLearningConfig(unittest.TestCase):
    """Test continual learning configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ContinualLearningConfig()
    
    def test_cl_config_defaults(self):
        """Test default continual learning configuration."""
        self.assertGreater(self.config.ewc_lambda, 0)
        self.assertGreater(self.config.fisher_damping, 0)


class TestFCLConfig(unittest.TestCase):
    """Test main FCL configuration combining all modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FCLConfig()
    
    def test_fcl_config_has_all_components(self):
        """Test FCL config has all required components."""
        self.assertIsNotNone(self.config.model)
        self.assertIsNotNone(self.config.training)
        self.assertIsNotNone(self.config.der)
        self.assertIsNotNone(self.config.multimodal)
        self.assertIsNotNone(self.config.cl)
    
    def test_fcl_config_component_types(self):
        """Test component types are correct."""
        self.assertIsInstance(self.config.model, ModelConfig)
        self.assertIsInstance(self.config.training, TrainingConfig)
        self.assertIsInstance(self.config.der, DERConfig)
        self.assertIsInstance(self.config.multimodal, MultimodalConfig)
        self.assertIsInstance(self.config.cl, ContinualLearningConfig)
    
    def test_fcl_config_consistency(self):
        """Test consistency of configurations."""
        # Input/output dimensions should match
        self.assertEqual(
            self.config.model.input_dim,
            13  # Clinical features
        )
        self.assertEqual(
            self.config.model.output_dim,
            2   # Binary classification
        )
    
    def test_fcl_config_batch_size_consistency(self):
        """Test batch size is valid for DER buffer."""
        self.assertLessEqual(
            self.config.training.batch_size,
            self.config.der.buffer_size
        )
    
    def test_fcl_config_validation(self):
        """Test entire FCL config can be validated."""
        # Should not raise error
        self.config.der.validate()
        
        # All components should be initialized
        self.assertIsNotNone(self.config.model)
        self.assertIsNotNone(self.config.training)


class TestConfigIntegration(unittest.TestCase):
    """Test configuration integration across modules."""
    
    def test_multimodal_feature_dims_match(self):
        """Test multimodal config dimensions are compatible."""
        config = FCLConfig()
        
        # Image features from CNN + tabular features from Transformer
        image_dim = config.multimodal.feature_dim
        tabular_dim = config.model.token_dim
        
        # Combined dimension should be > 0
        combined_dim = image_dim + tabular_dim
        self.assertGreater(combined_dim, 0)
    
    def test_der_config_buffer_bounds(self):
        """Test DER buffer configuration bounds."""
        config = FCLConfig()
        
        # Buffer should be large enough for at least one batch
        self.assertGreaterEqual(
            config.der.buffer_size,
            config.training.batch_size
        )
    
    def test_config_serialization(self):
        """Test configs can be converted to dictionaries."""
        config = FCLConfig()
        
        # Get DER config as dict
        der_dict = config.der.get_config_dict()
        
        self.assertIsInstance(der_dict, dict)
        self.assertGreater(len(der_dict), 0)


if __name__ == '__main__':
    unittest.main()
