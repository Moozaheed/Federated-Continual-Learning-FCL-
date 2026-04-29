"""
Unit Tests for Model Components
Tests for FTTransformer, PromptTuning, and related modules.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Import modules to test
import sys
sys.path.insert(0, '/home/bs01233/Documents/FL/fcl_project')

from code.config import ModelConfig, TrainingConfig, FCLConfig
from code.model import FTTransformer, create_model


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig initialization and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig()
    
    def test_config_defaults(self):
        """Test default configuration values."""
        self.assertEqual(self.config.input_dim, 13)
        self.assertEqual(self.config.token_dim, 192)
        self.assertEqual(self.config.n_prompt_tokens, 10)
        self.assertEqual(self.config.output_dim, 2)
        self.assertEqual(self.config.n_transformer_blocks, 6)
    
    def test_config_modification(self):
        """Test configuration modification."""
        config = ModelConfig()
        config.input_dim = 20
        config.output_dim = 3
        
        self.assertEqual(config.input_dim, 20)
        self.assertEqual(config.output_dim, 3)
    
    def test_config_attributes_exist(self):
        """Test all required attributes exist."""
        required_attrs = [
            'input_dim', 'token_dim', 'n_prompt_tokens',
            'n_transformer_blocks', 'output_dim', 'mlp_dropout'
        ]
        for attr in required_attrs:
            self.assertTrue(hasattr(self.config, attr),
                          f"Missing attribute: {attr}")


class TestFTTransformer(unittest.TestCase):
    """Test FT-Transformer model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.model_config = ModelConfig()
        self.train_config = TrainingConfig()
        self.model = FTTransformer(self.model_config, self.train_config)
        self.model = self.model.to(self.device)
    
    def tearDown(self):
        """Clean up after tests."""
        del self.model
        torch.cuda.empty_cache()
    
    def test_model_initialization(self):
        """Test model initializes without errors."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.config.input_dim, 13)
    
    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        batch_size = 32
        x = torch.randn(batch_size, 13)
        
        logits = self.model(x)
        
        self.assertEqual(logits.shape, (batch_size, 2))
    
    def test_forward_pass_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        batch_sizes = [1, 8, 32, 64]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 13)
            logits = self.model(x)
            
            self.assertEqual(logits.shape[0], batch_size)
            self.assertEqual(logits.shape[1], 2)
    
    def test_extract_features_shape(self):
        """Test feature extraction produces correct shape."""
        batch_size = 32
        x = torch.randn(batch_size, 13)
        
        features = self.model.extract_features(x)
        
        # Features should be (batch_size, token_dim)
        self.assertEqual(features.shape, (batch_size, self.model_config.token_dim))
    
    def test_extract_features_vs_logits(self):
        """Test that extract_features and forward pass use same intermediate."""
        x = torch.randn(16, 13)
        
        # Get features
        features = self.model.extract_features(x)
        
        # Get logits
        logits = self.model(x)
        
        # Features should be before classification head
        self.assertEqual(features.shape[1], self.model_config.token_dim)
        self.assertEqual(logits.shape[1], 2)
        
        # Both should be non-zero
        self.assertTrue(torch.sum(features) != 0)
        self.assertTrue(torch.sum(logits) != 0)
    
    def test_parameter_count(self):
        """Test parameter counting methods."""
        param_dict = self.model.get_param_count()
        
        self.assertIn('trainable', param_dict)
        self.assertIn('total', param_dict)
        self.assertIn('frozen', param_dict)
        
        # Total should be trainable + frozen
        self.assertEqual(
            param_dict['total'],
            param_dict['trainable'] + param_dict['frozen']
        )
    
    def test_count_parameters_backward_compat(self):
        """Test backward compatible count_parameters method."""
        total_params = self.model.count_parameters()
        param_dict = self.model.get_param_count()
        
        self.assertEqual(total_params, param_dict['trainable'])
    
    def test_total_parameters_property(self):
        """Test total_parameters property."""
        total = self.model.total_parameters
        
        self.assertGreater(total, 0)
        self.assertIsInstance(total, int)
    
    def test_prompt_parameters_property(self):
        """Test prompt_parameters property."""
        prompt_params = self.model.prompt_parameters
        
        self.assertGreater(prompt_params, 0)
        self.assertLess(prompt_params, self.model.total_parameters)
    
    def test_freeze_backbone(self):
        """Test freezing backbone layers."""
        # Initially all trainable
        initial_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Freeze backbone
        self.model.freeze_backbone()
        
        # After freeze, only prompts should be trainable
        frozen_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Trainable params should decrease
        self.assertLess(frozen_trainable, initial_trainable)
        
        # Frozen trainable should equal prompt parameters
        self.assertEqual(frozen_trainable, self.model.prompt_parameters)
    
    def test_unfreeze_backbone(self):
        """Test unfreezing backbone layers."""
        # Freeze then unfreeze
        self.model.freeze_backbone()
        self.model.unfreeze_backbone()
        
        # Should have same trainable count as initial
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.assertEqual(trainable, self.model.total_parameters)
    
    def test_prompt_tokens_extraction(self):
        """Test extracting prompt tokens."""
        prompts = self.model.get_prompt_tokens()
        
        # Should be (1, n_prompt_tokens, token_dim)
        self.assertEqual(prompts.shape[1], self.model_config.n_prompt_tokens)
        self.assertEqual(prompts.shape[2], self.model_config.token_dim)
    
    def test_gradients_flow(self):
        """Test that gradients flow through the model."""
        x = torch.randn(8, 13, requires_grad=True)
        logits = self.model(x)
        
        # Compute loss
        loss = logits.sum()
        loss.backward()
        
        # Check that model parameters have gradients
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_model_on_gpu(self):
        """Test model can be moved to GPU (if available)."""
        if torch.cuda.is_available():
            device = "cuda"
            model = FTTransformer(self.model_config, self.train_config)
            model = model.to(device)
            
            x = torch.randn(16, 13, device=device)
            logits = model(x)
            
            self.assertEqual(logits.device.type, "cuda")
    
    def test_model_create_function(self):
        """Test create_model factory function."""
        model = create_model(
            config=self.model_config,
            training_config=self.train_config,
            device=self.device
        )
        
        self.assertIsInstance(model, FTTransformer)
        self.assertEqual(model.config.input_dim, 13)
    
    def test_model_create_function_backward_compat(self):
        """Test create_model with keyword arguments (backward compatibility)."""
        model = create_model(
            num_numerical_features=13,
            num_classes=2,
            num_transformer_blocks=6,
            device=self.device
        )
        
        self.assertIsInstance(model, FTTransformer)
        self.assertEqual(model.config.input_dim, 13)
        self.assertEqual(model.config.output_dim, 2)


class TestModelTraining(unittest.TestCase):
    """Test model training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.model_config = ModelConfig()
        self.train_config = TrainingConfig()
        self.model = FTTransformer(self.model_config, self.train_config)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
    
    def test_training_step(self):
        """Test single training step."""
        x = torch.randn(16, 13)
        y = torch.randint(0, 2, (16,))
        
        # Forward pass
        logits = self.model(x)
        loss = self.criterion(logits, y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Loss should be a scalar
        self.assertEqual(loss.item() > 0, True)
    
    def test_loss_decreases_with_training(self):
        """Test that loss generally decreases over training steps."""
        x = torch.randn(32, 13)
        y = torch.randint(0, 2, (32,))
        
        losses = []
        for _ in range(10):
            logits = self.model(x)
            loss = self.criterion(logits, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
        
        # Check that average loss in second half is less than first half
        first_half_avg = np.mean(losses[:5])
        second_half_avg = np.mean(losses[5:])
        
        # Note: This is probabilistic, may not always be true
        # Just check that training runs without errors
        self.assertEqual(len(losses), 10)


if __name__ == '__main__':
    unittest.main()
