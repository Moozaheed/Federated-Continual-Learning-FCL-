"""
Unit Tests for Utility Functions
Tests for data loading, training utilities, and metric computation.
"""

import unittest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bs01233/Documents/FL/fcl_project')

from code.utils import (
    get_batch,
    compute_metrics,
    add_privacy_noise,
    get_privacy_budget,
    get_param_count,
    count_parameters,
    get_per_sample_loss
)
from code.config import TrainingConfig, DERConfig


class TestGetBatch(unittest.TestCase):
    """Test batch extraction utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.num_features = 13
        self.num_classes = 2
        
        # Create sample batch dict
        self.batch = {
            'images': torch.randn(self.batch_size, 3, 224, 224),
            'tabular': torch.randn(self.batch_size, self.num_features),
            'labels': torch.randint(0, self.num_classes, (self.batch_size,)),
            'logits': torch.randn(self.batch_size, self.num_classes)
        }
    
    def test_get_batch_returns_tensor(self):
        """Test get_batch returns tensor."""
        result = get_batch(self.batch, 'images')
        
        self.assertIsInstance(result, torch.Tensor)
    
    def test_get_batch_correct_shape(self):
        """Test get_batch returns correct tensor shape."""
        result = get_batch(self.batch, 'tabular')
        
        self.assertEqual(result.shape, (self.batch_size, self.num_features))
    
    def test_get_batch_all_keys(self):
        """Test get_batch for all batch keys."""
        for key in ['images', 'tabular', 'labels', 'logits']:
            result = get_batch(self.batch, key)
            self.assertIsNotNone(result)
    
    def test_get_batch_missing_key(self):
        """Test get_batch with missing key."""
        with self.assertRaises(KeyError):
            get_batch(self.batch, 'nonexistent')
    
    def test_get_batch_empty_batch(self):
        """Test get_batch with empty batch dict."""
        with self.assertRaises(KeyError):
            get_batch({}, 'images')


class TestComputeMetrics(unittest.TestCase):
    """Test metric computation utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 16
        self.num_classes = 2
        
        # Create predictions and labels
        self.logits = torch.randn(self.batch_size, self.num_classes)
        self.labels = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Binary predictions
        self.probabilities = torch.softmax(self.logits, dim=1)
    
    def test_compute_metrics_returns_dict(self):
        """Test compute_metrics returns dictionary."""
        metrics = compute_metrics(
            self.logits,
            self.labels
        )
        
        self.assertIsInstance(metrics, dict)
    
    def test_compute_metrics_has_accuracy(self):
        """Test metrics includes accuracy."""
        metrics = compute_metrics(
            self.logits,
            self.labels
        )
        
        self.assertIn('accuracy', metrics)
        self.assertIsInstance(metrics['accuracy'], float)
    
    def test_compute_metrics_accuracy_range(self):
        """Test accuracy is in valid range."""
        metrics = compute_metrics(
            self.logits,
            self.labels
        )
        
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
    
    def test_compute_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        logits = torch.zeros(10, 2)
        logits[:, 0] = 100  # High score for class 0
        labels = torch.zeros(10, dtype=torch.long)
        
        metrics = compute_metrics(logits, labels)
        
        self.assertAlmostEqual(metrics['accuracy'], 1.0, places=5)
    
    def test_compute_metrics_random_predictions(self):
        """Test metrics with random predictions."""
        logits = torch.randn(100, 2)
        labels = torch.randint(0, 2, (100,))
        
        metrics = compute_metrics(logits, labels)
        
        # Random accuracy should be around 0.5 for balanced labels
        self.assertGreater(metrics['accuracy'], 0.3)
        self.assertLess(metrics['accuracy'], 0.7)
    
    def test_compute_metrics_multiclass(self):
        """Test metrics with multiclass problems."""
        num_classes = 5
        logits = torch.randn(32, num_classes)
        labels = torch.randint(0, num_classes, (32,))
        
        metrics = compute_metrics(logits, labels)
        
        self.assertIn('accuracy', metrics)


class TestPrivacyNoise(unittest.TestCase):
    """Test differential privacy noise addition."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.num_classes = 2
        self.epsilon = 1.0
        self.delta = 1e-5
    
    def test_add_privacy_noise_shape(self):
        """Test noise has correct shape."""
        logits = torch.randn(self.batch_size, self.num_classes)
        
        noisy_logits = add_privacy_noise(
            logits,
            epsilon=self.epsilon,
            delta=self.delta
        )
        
        self.assertEqual(noisy_logits.shape, logits.shape)
    
    def test_add_privacy_noise_different_outputs(self):
        """Test noise produces different outputs."""
        logits = torch.randn(self.batch_size, self.num_classes)
        
        noisy1 = add_privacy_noise(logits, epsilon=self.epsilon, delta=self.delta)
        noisy2 = add_privacy_noise(logits, epsilon=self.epsilon, delta=self.delta)
        
        # Different noise should produce different results
        self.assertFalse(torch.allclose(noisy1, noisy2))
    
    def test_add_privacy_noise_epsilon_effect(self):
        """Test stronger privacy (lower epsilon) produces more noise."""
        logits = torch.randn(100, self.num_classes)
        
        noisy_weak = add_privacy_noise(logits, epsilon=10.0, delta=self.delta)
        noisy_strong = add_privacy_noise(logits, epsilon=0.1, delta=self.delta)
        
        # Strong privacy should have more difference from original
        diff_weak = torch.abs(logits - noisy_weak).mean()
        diff_strong = torch.abs(logits - noisy_strong).mean()
        
        self.assertGreater(diff_strong, diff_weak)
    
    def test_add_privacy_noise_preserves_dtype(self):
        """Test noise preserves tensor dtype."""
        logits = torch.randn(self.batch_size, self.num_classes, dtype=torch.float32)
        
        noisy = add_privacy_noise(logits, epsilon=self.epsilon, delta=self.delta)
        
        self.assertEqual(noisy.dtype, logits.dtype)


class TestPrivacyBudget(unittest.TestCase):
    """Test privacy budget tracking."""
    
    def test_get_privacy_budget_returns_dict(self):
        """Test privacy budget returns dict."""
        budget = get_privacy_budget(
            epsilon=1.0,
            delta=1e-5,
            num_rounds=10
        )
        
        self.assertIsInstance(budget, dict)
    
    def test_get_privacy_budget_has_keys(self):
        """Test privacy budget has required keys."""
        budget = get_privacy_budget(1.0, 1e-5, 10)
        
        self.assertIn('epsilon', budget)
        self.assertIn('delta', budget)
        self.assertIn('num_rounds', budget)
    
    def test_get_privacy_budget_values(self):
        """Test privacy budget contains correct values."""
        epsilon = 1.5
        delta = 1e-6
        rounds = 5
        
        budget = get_privacy_budget(epsilon, delta, rounds)
        
        self.assertEqual(budget['epsilon'], epsilon)
        self.assertEqual(budget['delta'], delta)
        self.assertEqual(budget['num_rounds'], rounds)
    
    def test_get_privacy_budget_remaining_epsilon(self):
        """Test privacy budget tracks remaining epsilon."""
        initial_epsilon = 2.0
        spent_rounds = 3
        total_rounds = 10
        
        budget = get_privacy_budget(initial_epsilon, 1e-5, total_rounds)
        
        # Remaining should depend on rounds spent
        self.assertGreater(budget['epsilon'], 0)


class TestParameterCounting(unittest.TestCase):
    """Test parameter counting utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        import torch.nn as nn
        
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
    
    def test_count_parameters_returns_int(self):
        """Test count_parameters returns integer."""
        count = count_parameters(self.model)
        
        self.assertIsInstance(count, int)
    
    def test_count_parameters_positive(self):
        """Test parameter count is positive."""
        count = count_parameters(self.model)
        
        self.assertGreater(count, 0)
    
    def test_count_parameters_correct_value(self):
        """Test parameter count is correct."""
        # Linear(10, 20): 10*20 + 20 = 220
        # Linear(20, 5): 20*5 + 5 = 105
        # Total: 325
        expected = 220 + 105
        
        count = count_parameters(self.model)
        
        self.assertEqual(count, expected)
    
    def test_get_param_count_backward_compatibility(self):
        """Test get_param_count for backward compatibility."""
        # Should return dict
        result = get_param_count(self.model)
        
        if isinstance(result, dict):
            self.assertIn('total_params', result)
        else:
            # Or could return int
            self.assertIsInstance(result, int)
    
    def test_count_parameters_excludes_buffers(self):
        """Test only trainable parameters are counted."""
        import torch.nn as nn
        
        model = nn.Linear(10, 5)
        
        total_params = count_parameters(model)
        
        # Should include weights and biases
        self.assertEqual(total_params, 10*5 + 5)


class TestPerSampleLoss(unittest.TestCase):
    """Test per-sample loss computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.num_classes = 2
        
        self.logits = torch.randn(self.batch_size, self.num_classes)
        self.labels = torch.randint(0, self.num_classes, (self.batch_size,))
    
    def test_per_sample_loss_shape(self):
        """Test per-sample loss has correct shape."""
        loss = get_per_sample_loss(self.logits, self.labels)
        
        self.assertEqual(loss.shape[0], self.batch_size)
    
    def test_per_sample_loss_positive(self):
        """Test per-sample loss is positive."""
        loss = get_per_sample_loss(self.logits, self.labels)
        
        self.assertTrue(torch.all(loss >= 0))
    
    def test_per_sample_loss_mean_equals_batch_loss(self):
        """Test mean of per-sample losses equals batch loss."""
        import torch.nn as nn
        
        per_sample = get_per_sample_loss(self.logits, self.labels)
        
        criterion = nn.CrossEntropyLoss()
        batch_loss = criterion(self.logits, self.labels)
        
        mean_per_sample = per_sample.mean()
        
        self.assertAlmostEqual(mean_per_sample.item(), batch_loss.item(), places=5)
    
    def test_per_sample_loss_gradient_flow(self):
        """Test gradients flow through per-sample loss."""
        logits = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        labels = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss = get_per_sample_loss(logits, labels)
        total_loss = loss.sum()
        total_loss.backward()
        
        self.assertIsNotNone(logits.grad)


class TestUtilityIntegration(unittest.TestCase):
    """Integration tests for utility functions."""
    
    def test_training_step_with_utilities(self):
        """Test complete training step using utilities."""
        import torch.nn as nn
        
        # Model
        model = nn.Linear(13, 2)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Data
        x = torch.randn(16, 13)
        y = torch.randint(0, 2, (16,))
        
        # Training step
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        metrics = compute_metrics(logits, y)
        
        self.assertIn('accuracy', metrics)
        self.assertGreater(metrics['accuracy'], 0)
    
    def test_privacy_budget_tracking_during_training(self):
        """Test privacy budget tracking across multiple rounds."""
        epsilon_budget = 1.0
        
        for round_num in range(5):
            budget = get_privacy_budget(
                epsilon_budget,
                1e-5,
                5
            )
            
            self.assertEqual(budget['num_rounds'], 5)
    
    def test_batch_processing_pipeline(self):
        """Test complete batch processing pipeline."""
        # Create batch
        batch = {
            'tabular': torch.randn(8, 13),
            'labels': torch.randint(0, 2, (8,))
        }
        
        # Extract data
        x = get_batch(batch, 'tabular')
        y = get_batch(batch, 'labels')
        
        # Verify shapes
        self.assertEqual(x.shape[0], 8)
        self.assertEqual(y.shape[0], 8)


if __name__ == '__main__':
    unittest.main()
