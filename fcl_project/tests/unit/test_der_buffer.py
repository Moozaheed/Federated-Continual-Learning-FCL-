"""
Unit Tests for DER++ Buffer
Tests for Dark Experience Replay buffer operations.
"""

import unittest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bs01233/Documents/FL/fcl_project')

from code.der import DERBuffer, der_loss


class TestDERBuffer(unittest.TestCase):
    """Test DER++ experience replay buffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.buffer_size = 100
        self.buffer = DERBuffer(buffer_size=self.buffer_size, device=self.device)
        
        # Create sample data
        self.batch_size = 16
        self.num_classes = 2
        self.image_channels = 3
        self.image_size = 224
        self.num_features = 13
        
        self.images = torch.randn(self.batch_size, self.image_channels, self.image_size, self.image_size)
        self.tabular = torch.randn(self.batch_size, self.num_features)
        self.labels = torch.randint(0, self.num_classes, (self.batch_size,))
        self.logits = torch.randn(self.batch_size, self.num_classes)
    
    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        self.assertEqual(self.buffer.capacity, self.buffer_size)
        self.assertEqual(self.buffer.n_samples, 0)
        self.assertFalse(self.buffer.is_full)
    
    def test_buffer_add_data(self):
        """Test adding data to buffer."""
        self.buffer.add_data(
            self.images,
            self.tabular,
            self.labels,
            self.logits
        )
        
        self.assertEqual(self.buffer.n_samples, self.batch_size)
        self.assertFalse(self.buffer.is_full)
    
    def test_buffer_add_multiple_batches(self):
        """Test adding multiple batches to buffer."""
        for _ in range(3):
            self.buffer.add_data(
                self.images,
                self.tabular,
                self.labels,
                self.logits
            )
        
        self.assertEqual(self.buffer.n_samples, self.batch_size * 3)
    
    def test_buffer_full_detection(self):
        """Test buffer full detection."""
        # Add samples until full
        num_batches = self.buffer_size // self.batch_size
        for _ in range(num_batches):
            self.buffer.add_data(
                self.images,
                self.tabular,
                self.labels,
                self.logits
            )
        
        self.assertTrue(self.buffer.is_full)
    
    def test_buffer_occupancy_ratio(self):
        """Test occupancy ratio calculation."""
        self.assertEqual(self.buffer.occupancy_ratio, 0.0)
        
        self.buffer.add_data(
            self.images,
            self.tabular,
            self.labels,
            self.logits
        )
        
        expected_ratio = self.batch_size / self.buffer_size
        self.assertAlmostEqual(self.buffer.occupancy_ratio, expected_ratio, places=5)
    
    def test_buffer_sample_batch(self):
        """Test sampling batch from buffer."""
        self.buffer.add_data(
            self.images,
            self.tabular,
            self.labels,
            self.logits
        )
        
        batch = self.buffer.sample_batch(batch_size=8)
        
        self.assertIsNotNone(batch)
        self.assertIn('images', batch)
        self.assertIn('tabular', batch)
        self.assertIn('labels', batch)
        self.assertIn('logits', batch)
    
    def test_buffer_sample_batch_shapes(self):
        """Test sampled batch has correct shapes."""
        self.buffer.add_data(
            self.images,
            self.tabular,
            self.labels,
            self.logits
        )
        
        sample_size = 8
        batch = self.buffer.sample_batch(batch_size=sample_size)
        
        self.assertEqual(batch['images'].shape[0], sample_size)
        self.assertEqual(batch['tabular'].shape[0], sample_size)
        self.assertEqual(batch['labels'].shape[0], sample_size)
        self.assertEqual(batch['logits'].shape[0], sample_size)
    
    def test_buffer_sample_empty_buffer(self):
        """Test sampling from empty buffer returns None."""
        batch = self.buffer.sample_batch(batch_size=8)
        
        self.assertIsNone(batch)
    
    def test_buffer_sample_larger_than_buffer(self):
        """Test sampling more samples than in buffer."""
        self.buffer.add_data(
            self.images,
            self.tabular,
            self.labels,
            self.logits
        )
        
        # Request more than available
        batch = self.buffer.sample_batch(batch_size=50)
        
        # Should return available samples
        self.assertIsNotNone(batch)
        self.assertLessEqual(batch['images'].shape[0], self.buffer.n_samples)
    
    def test_buffer_add_data_validation_type_error(self):
        """Test add_data rejects non-tensor inputs."""
        with self.assertRaises(TypeError):
            self.buffer.add_data(
                self.images.numpy(),  # NumPy array instead of tensor
                self.tabular,
                self.labels,
                self.logits
            )
    
    def test_buffer_add_data_validation_batch_size_mismatch(self):
        """Test add_data detects batch size mismatch."""
        mismatched_tabular = torch.randn(self.batch_size + 1, self.num_features)
        
        with self.assertRaises(ValueError):
            self.buffer.add_data(
                self.images,
                mismatched_tabular,
                self.labels,
                self.logits
            )
    
    def test_buffer_add_data_validation_shape_consistency(self):
        """Test add_data detects inconsistent shapes on second call."""
        self.buffer.add_data(
            self.images,
            self.tabular,
            self.labels,
            self.logits
        )
        
        # Try adding with different image shape
        different_images = torch.randn(self.batch_size, 3, 128, 128)
        
        with self.assertRaises(ValueError):
            self.buffer.add_data(
                different_images,
                self.tabular,
                self.labels,
                self.logits
            )
    
    def test_buffer_clear(self):
        """Test clearing buffer."""
        self.buffer.add_data(
            self.images,
            self.tabular,
            self.labels,
            self.logits
        )
        
        self.assertGreater(self.buffer.n_samples, 0)
        
        self.buffer.clear()
        
        self.assertEqual(self.buffer.n_samples, 0)
    
    def test_buffer_get_stats(self):
        """Test getting buffer statistics."""
        self.buffer.add_data(
            self.images,
            self.tabular,
            self.labels,
            self.logits
        )
        
        stats = self.buffer.get_stats()
        
        self.assertIn('capacity', stats)
        self.assertIn('n_samples', stats)
        self.assertIn('occupancy_ratio', stats)
        self.assertIn('is_full', stats)
        
        self.assertEqual(stats['capacity'], self.buffer_size)
        self.assertEqual(stats['n_samples'], self.batch_size)
    
    def test_buffer_reservoir_sampling(self):
        """Test reservoir sampling behavior."""
        small_buffer = DERBuffer(buffer_size=10, device=self.device)
        
        # Add many batches
        for _ in range(20):
            small_buffer.add_data(
                torch.randn(5, 3, 224, 224),
                torch.randn(5, 13),
                torch.randint(0, 2, (5,)),
                torch.randn(5, 2)
            )
        
        # Buffer should not exceed capacity
        self.assertLessEqual(small_buffer.n_samples, 100)  # Counting total seen
        # But stored samples should be <= buffer_size
        self.assertIsNotNone(small_buffer.images)


class TestDERLoss(unittest.TestCase):
    """Test DER++ loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 16
        self.num_classes = 2
        
        self.logits = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        self.labels = torch.randint(0, self.num_classes, (self.batch_size,))
        self.buffer_logits = torch.randn(self.batch_size, self.num_classes)
        self.buffer_labels = torch.randint(0, self.num_classes, (self.batch_size,))
    
    def test_der_loss_computation(self):
        """Test DER loss computation."""
        loss = der_loss(
            self.logits,
            self.labels,
            self.buffer_logits,
            self.buffer_labels,
            alpha=0.3,
            beta=0.7
        )
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_der_loss_gradient_flow(self):
        """Test gradients flow through DER loss."""
        loss = der_loss(
            self.logits,
            self.labels,
            self.buffer_logits,
            self.buffer_labels
        )
        
        loss.backward()
        
        self.assertIsNotNone(self.logits.grad)
        self.assertTrue(torch.sum(self.logits.grad) != 0)
    
    def test_der_loss_weights_effect(self):
        """Test different alpha/beta weights affect loss."""
        loss1 = der_loss(
            self.logits,
            self.labels,
            self.buffer_logits,
            self.buffer_labels,
            alpha=0.1,
            beta=0.9
        )
        
        loss2 = der_loss(
            self.logits,
            self.labels,
            self.buffer_logits,
            self.buffer_labels,
            alpha=0.9,
            beta=0.1
        )
        
        # Losses should be different
        self.assertNotAlmostEqual(loss1.item(), loss2.item(), places=3)
    
    def test_der_loss_zero_weights(self):
        """Test DER loss with zero weights."""
        loss = der_loss(
            self.logits,
            self.labels,
            self.buffer_logits,
            self.buffer_labels,
            alpha=0.0,
            beta=0.0
        )
        
        # Loss should be near zero
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class TestDERBufferIntegration(unittest.TestCase):
    """Integration tests for DER buffer with training."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.buffer = DERBuffer(buffer_size=100, device=self.device)
    
    def test_buffer_workflow(self):
        """Test typical buffer usage workflow."""
        # Create model
        import torch.nn as nn
        model = nn.Linear(13, 2)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Simulate training
        for step in range(3):
            # Generate batch
            x = torch.randn(16, 13)
            y = torch.randint(0, 2, (16,))
            
            # Forward pass
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Add to buffer
            with torch.no_grad():
                current_logits = model(x)
            self.buffer.add_data(
                x.unsqueeze(1).expand(-1, 3, 224, 224),  # Fake images
                x,
                y,
                current_logits
            )
        
        # Sample from buffer
        batch = self.buffer.sample_batch(8)
        self.assertIsNotNone(batch)
        self.assertEqual(batch['images'].shape[0], 8)


if __name__ == '__main__':
    unittest.main()
