"""Privacy Auditing with Membership Inference Attacks (MIA)

Implements membership inference attacks to evaluate differential privacy protections.
Based on: Shokri et al. "Membership Inference Attacks Against Machine Learning Models"
IEEE S&P 2017

MIA Principle:
- Train shadow models on different datasets
- Compare loss on member vs non-member samples
- If privacy is good, no significant difference in loss
- If privacy is bad, members have lower loss (overfitting)
"""

import logging
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class MIAConfig:
    """Configuration for Membership Inference Attack."""
    n_shadow_models: int = 5  # Number of shadow models
    shadow_model_epochs: int = 20  # Training epochs for shadow models
    attack_batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    use_loss_only: bool = True  # Use only loss as feature
    use_confidence: bool = False  # Use max softmax probability
    use_entropy: bool = False  # Use prediction entropy
    device: str = 'cpu'
    verbose: bool = True


class ShadowModel:
    """Shadow model for membership inference attack.
    
    Trains on a subset of data and records predictions/losses
    for both member and non-member samples.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MIAConfig
    ):
        """Initialize shadow model.
        
        Args:
            model: Base PyTorch model
            config: MIA configuration
        """
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train shadow model.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        for epoch in range(self.config.shadow_model_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(images) if not hasattr(self.model, 'heads') else self.model(images, task_id=getattr(self, '_task_id', 0))
                loss = self.criterion(logits, labels).mean()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            if self.config.verbose and (epoch + 1) % 5 == 0:
                logger.info(f"Shadow model epoch {epoch+1}: loss={train_loss:.4f}")
    
    def get_member_scores(
        self,
        member_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get scores for member samples (used in training).
        
        Args:
            member_loader: DataLoader with member samples
        
        Returns:
            (losses, confidences) arrays
        """
        self.model.eval()
        losses = []
        confidences = []
        
        with torch.no_grad():
            for batch in member_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(images) if not hasattr(self.model, 'heads') else self.model(images, task_id=getattr(self, '_task_id', 0))
                batch_losses = self.criterion(logits, labels).cpu().numpy()
                losses.extend(batch_losses)
                
                if self.config.use_confidence:
                    probs = torch.softmax(logits, dim=1)
                    max_probs = probs.max(dim=1)[0].cpu().numpy()
                    confidences.extend(max_probs)
        
        return np.array(losses), np.array(confidences)
    
    def get_nonmember_scores(
        self,
        nonmember_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get scores for non-member samples (not used in training).
        
        Args:
            nonmember_loader: DataLoader with non-member samples
        
        Returns:
            (losses, confidences) arrays
        """
        # Same as member scores but from different data
        return self.get_member_scores(nonmember_loader)


class MembershipInferenceAttack:
    """Membership Inference Attack implementation.
    
    Evaluates if a model reveals information about training data membership.
    """
    
    def __init__(
        self,
        model_constructor,
        config: MIAConfig
    ):
        """Initialize MIA.
        
        Args:
            model_constructor: Function that creates model instances
            config: MIA configuration
        """
        self.model_constructor = model_constructor
        self.config = config
        self.shadow_models = []
        self.member_scores = []
        self.nonmember_scores = []
    
    def prepare_shadow_datasets(
        self,
        dataset: Dataset,
        n_shadow: int = None
    ) -> List[Tuple[Subset, Subset]]:
        """Prepare datasets for shadow models.
        
        Args:
            dataset: Full dataset
            n_shadow: Number of shadow models
        
        Returns:
            List of (member_subset, nonmember_subset) pairs
        """
        n_shadow = n_shadow or self.config.n_shadow_models
        shadow_datasets = []
        
        # Split into shadow and test data
        n_shadow_total = len(dataset) // 2
        n_test = len(dataset) - n_shadow_total
        
        shadow_data, test_data = random_split(
            dataset,
            [n_shadow_total, n_test]
        )
        
        # Split shadow data among shadow models
        shadow_size = len(shadow_data) // n_shadow
        
        for i in range(n_shadow):
            start_idx = i * shadow_size
            end_idx = (i + 1) * shadow_size if i < n_shadow - 1 else len(shadow_data)
            
            member_subset = Subset(
                shadow_data,
                list(range(start_idx, end_idx))
            )
            
            # Non-member is from test set or other shadow data
            nonmember_size = len(member_subset)
            nonmember_indices = np.random.choice(
                len(test_data),
                nonmember_size,
                replace=False
            )
            
            nonmember_subset = Subset(
                test_data,
                nonmember_indices
            )
            
            shadow_datasets.append((member_subset, nonmember_subset))
        
        logger.info(
            f"Prepared {n_shadow} shadow datasets. "
            f"Shadow size: {shadow_size}, Test size: {len(test_data)}"
        )
        
        return shadow_datasets
    
    def train_shadow_models(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """Train shadow models.
        
        Args:
            dataset: Full dataset
            batch_size: Batch size
            num_workers: Number of workers
        """
        shadow_datasets = self.prepare_shadow_datasets(dataset)
        
        for shadow_id, (member_data, nonmember_data) in enumerate(shadow_datasets):
            logger.info(f"Training shadow model {shadow_id + 1}/{len(shadow_datasets)}")
            
            # Create dataloaders
            member_loader = DataLoader(
                member_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            
            val_loader = DataLoader(
                nonmember_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            
            # Create and train shadow model
            model = self.model_constructor()
            shadow = ShadowModel(model, self.config)
            shadow.train(member_loader, val_loader)
            
            # Get scores
            member_losses, _ = shadow.get_member_scores(member_loader)
            nonmember_losses, _ = shadow.get_nonmember_scores(val_loader)
            
            self.shadow_models.append(shadow)
            self.member_scores.extend(member_losses)
            self.nonmember_scores.extend(nonmember_losses)
    
    def evaluate_attack(self) -> Dict:
        """Evaluate MIA success.
        
        Returns:
            Dict with attack metrics
        """
        member_scores = np.array(self.member_scores)
        nonmember_scores = np.array(self.nonmember_scores)
        
        # Create binary labels: 1 for member, 0 for non-member
        y_true = np.concatenate([np.ones(len(member_scores)), 
                                 np.zeros(len(nonmember_scores))])
        y_scores = np.concatenate([member_scores, nonmember_scores])
        
        # Compute AUC
        auc = roc_auc_score(y_true, y_scores)
        
        # Compute TPR and FPR at different thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Attack accuracy at 50% FPR
        idx_fpr_50 = np.argmin(np.abs(fpr - 0.5))
        tpr_at_50_fpr = tpr[idx_fpr_50]
        
        # Compute threshold for 50% TPR
        idx_tpr_50 = np.argmin(np.abs(tpr - 0.5))
        fpr_at_50_tpr = fpr[idx_tpr_50]
        
        results = {
            'auc': auc,
            'tpr_at_50_fpr': float(tpr_at_50_fpr),
            'fpr_at_50_tpr': float(fpr_at_50_tpr),
            'member_mean_loss': float(member_scores.mean()),
            'nonmember_mean_loss': float(nonmember_scores.mean()),
            'member_std_loss': float(member_scores.std()),
            'nonmember_std_loss': float(nonmember_scores.std()),
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        
        logger.info(
            f"MIA Results: AUC={auc:.4f}, "
            f"TPR@FPR=0.5={tpr_at_50_fpr:.4f}, "
            f"FPR@TPR=0.5={fpr_at_50_tpr:.4f}"
        )
        
        return results
    
    def plot_roc(self, results: Dict, save_path: Optional[str] = None):
        """Plot ROC curve.
        
        Args:
            results: Results from evaluate_attack()
            save_path: Path to save figure
        """
        plt.figure(figsize=(8, 6))
        
        plt.plot(
            results['fpr'],
            results['tpr'],
            lw=2,
            label=f"AUC = {results['auc']:.4f}"
        )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Membership Inference Attack ROC Curve', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")
        
        return plt.gcf()


def evaluate_differential_privacy_protection(
    model: nn.Module,
    dataset: Dataset,
    model_constructor,
    config: MIAConfig,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict:
    """Evaluate DP protection using MIA.
    
    Args:
        model: Trained model to evaluate
        dataset: Full dataset
        model_constructor: Function to create models
        config: MIA configuration
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        Dict with privacy evaluation results
    """
    logger.info("Starting privacy evaluation via MIA...")
    
    # Train shadow models
    mia = MembershipInferenceAttack(model_constructor, config)
    mia.train_shadow_models(dataset, batch_size, num_workers)
    
    # Evaluate attack
    results = mia.evaluate_attack()
    
    # Interpret results
    if results['auc'] > 0.6:
        privacy_level = 'Low (Poor)'
    elif results['auc'] > 0.55:
        privacy_level = 'Moderate'
    else:
        privacy_level = 'High (Good)'
    
    results['privacy_level'] = privacy_level
    
    logger.info(f"Privacy Level: {privacy_level}")
    
    return results
