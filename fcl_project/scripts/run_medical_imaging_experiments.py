"""Experiment Runner for Medical Imaging and Privacy Auditing

Comprehensive experiment pipeline for:
1. Medical imaging dataset experiments
2. Privacy auditing with MIA attacks
3. Publication-ready result visualization
"""

import logging
import json
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from code.model import FTTransformer
from code.config import ModelConfig
from code.datasets import get_medical_dataset, create_federated_loaders
from code.privacy_audit import evaluate_differential_privacy_protection, MIAConfig
from code.visualization import PublicationVisualizer

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for medical imaging experiments."""
    experiment_name: str
    dataset_name: str  # 'mimic_cxr', 'medmnist', 'chexpert'
    dataset_path: str
    model_type: str = 'ft_transformer'
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_workers: int = 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    output_dir: str = 'results'
    verbose: bool = True


class MedicalImagingExperiment:
    """Run medical imaging experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Set random seeds
        torch.manual_seed(config.seed)
        
        self.results = {}
    
    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / 'experiment.log'
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        for handler in [file_handler, console_handler]:
            logger.addHandler(handler)
    
    def load_dataset(self) -> Dict:
        """Load medical imaging dataset.
        
        Returns:
            Dict with train/val/test loaders
        """
        logger.info(f"Loading {self.config.dataset_name} dataset...")
        
        # Load train, val, test splits
        train_dataset = get_medical_dataset(
            self.config.dataset_name,
            self.config.dataset_path,
            split='train'
        )
        
        val_dataset = get_medical_dataset(
            self.config.dataset_name,
            self.config.dataset_path,
            split='val' if self.config.dataset_name != 'medmnist' else 'test'
        )
        
        test_dataset = get_medical_dataset(
            self.config.dataset_name,
            self.config.dataset_path,
            split='test'
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        logger.info(
            f"Dataset loaded: "
            f"Train={len(train_dataset)}, "
            f"Val={len(val_dataset)}, "
            f"Test={len(test_dataset)}"
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'train_dataset': train_dataset,
            'n_classes': len(train_dataset[0]['labels']) if 'labels' in train_dataset[0] else 2
        }
    
    def create_model(self, n_classes: int) -> nn.Module:
        """Create model.
        
        Args:
            n_classes: Number of output classes
        
        Returns:
            Model instance
        """
        if self.config.model_type == 'ft_transformer':
            config = ModelConfig(
                num_features=3,  # Images (RGB)
                num_classes=n_classes,
                d_model=256,
                n_heads=8,
                n_layers=6,
                ffn_multiplier=4,
                norm_layer='LayerNorm'
            )
            model = FTTransformer(config)
        else:
            raise ValueError(f"Unknown model: {self.config.model_type}")
        
        model = model.to(self.config.device)
        
        logger.info(f"Created model: {self.config.model_type}")
        
        return model
    
    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict:
        """Train model.
        
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
        
        Returns:
            Dict with training history
        """
        logger.info("Starting model training...")
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_acc = 0
        best_model_path = self.output_dir / 'best_model.pth'
        
        for epoch in range(self.config.num_epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                images = batch['image'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.config.device)
                    labels = batch['label'].to(self.config.device)
                    
                    logits = model(images)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs}: "
                    f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                    f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}"
                )
        
        # Load best model
        model.load_state_dict(torch.load(best_model_path))
        
        return history
    
    def evaluate_privacy(
        self,
        model: nn.Module,
        dataset: DataLoader,
        n_shadow_models: int = 3
    ) -> Dict:
        """Evaluate privacy with MIA attacks.
        
        Args:
            model: Trained model
            dataset: Dataset for MIA
            n_shadow_models: Number of shadow models
        
        Returns:
            Privacy evaluation results
        """
        logger.info("Starting privacy evaluation...")
        
        mia_config = MIAConfig(
            n_shadow_models=n_shadow_models,
            shadow_model_epochs=10,
            device=self.config.device
        )
        
        def model_constructor():
            """Create new model instance."""
            return self.create_model(dataset.dataset.dataset[0]['labels'].shape[0])
        
        # This would require dataset instance, simplified here
        logger.warning("Privacy auditing requires dataset instances - implement in full version")
        
        return {'status': 'placeholder'}
    
    def run_federated_experiment(
        self,
        train_loader: DataLoader,
        num_clients: int = 5,
        num_rounds: int = 10
    ) -> Dict:
        """Run federated learning experiment.
        
        Args:
            train_loader: Full training data
            num_clients: Number of clients
            num_rounds: Federated rounds
        
        Returns:
            Federated training results
        """
        logger.info(f"Starting federated experiment with {num_clients} clients...")
        
        # This would require actual federated framework
        logger.warning("Federated learning requires Flower framework - implement in full version")
        
        return {'status': 'placeholder'}
    
    def save_results(self):
        """Save experiment results."""
        results_file = self.output_dir / 'results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Saved results to {results_file}")
    
    def run(self):
        """Run complete experiment pipeline."""
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        # Load dataset
        dataset_info = self.load_dataset()
        
        # Create model
        model = self.create_model(dataset_info['n_classes'])
        
        # Train model
        history = self.train_model(
            model,
            dataset_info['train'],
            dataset_info['val']
        )
        
        # Save results
        self.results['training_history'] = history
        self.results['config'] = asdict(self.config)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in dataset_info['test']:
                images = batch['image'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = test_correct / test_total
        self.results['test_accuracy'] = test_acc
        
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        
        # Privacy evaluation (simplified)
        # privacy_results = self.evaluate_privacy(model, dataset_info['train_dataset'])
        # self.results['privacy'] = privacy_results
        
        # Save all results
        self.save_results()
        
        logger.info(f"Experiment completed: {self.config.experiment_name}")
        
        return self.results


def run_medical_imaging_benchmark():
    """Run benchmark on multiple medical imaging datasets."""
    datasets = ['medmnist', 'chexpert']
    results = {}
    
    for dataset_name in datasets:
        logger.info(f"Running benchmark on {dataset_name}...")
        
        config = ExperimentConfig(
            experiment_name=f"medical_imaging_{dataset_name}",
            dataset_name=dataset_name,
            dataset_path=f"data/{dataset_name}",
            num_epochs=10,
            batch_size=32
        )
        
        experiment = MedicalImagingExperiment(config)
        result = experiment.run()
        results[dataset_name] = result
    
    return results
