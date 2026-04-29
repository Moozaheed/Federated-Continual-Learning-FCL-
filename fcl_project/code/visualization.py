"""Publication-Ready Visualizations

Creates high-quality, publication-grade visualizations (300 DPI) for research papers.
Includes ROC curves, confusion matrices, privacy-utility tradeoffs, and more.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

logger = logging.getLogger(__name__)

# Professional paper style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PublicationVisualizer:
    """Create publication-ready visualizations."""
    
    # Paper dimensions (inches)
    SINGLE_COL_WIDTH = 3.5  # Single column
    DOUBLE_COL_WIDTH = 7.0  # Double column (full page width)
    COL_HEIGHT = 2.5  # Column height
    DPI = 300  # Publication standard
    
    # Colors for consistency
    COLORS = {
        'primary': '#2E86C1',
        'secondary': '#E74C3C',
        'tertiary': '#27AE60',
        'quaternary': '#F39C12',
        'neutral': '#34495E'
    }
    
    @staticmethod
    def set_paper_style():
        """Set matplotlib style for publications."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['lines.markersize'] = 6
        plt.rcParams['figure.dpi'] = PublicationVisualizer.DPI
        plt.rcParams['savefig.dpi'] = PublicationVisualizer.DPI
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    @staticmethod
    def plot_roc_curves(
        results: Dict,
        title: str = "ROC Curves",
        save_path: Optional[str] = None,
        width: Optional[float] = None
    ) -> Tuple:
        """Plot ROC curves for multiple models.
        
        Args:
            results: Dict with model results, keys are model names
                Each value has 'fpr', 'tpr', 'auc'
            title: Plot title
            save_path: Path to save figure
            width: Figure width (default: SINGLE_COL_WIDTH)
        
        Returns:
            (figure, axis)
        """
        PublicationVisualizer.set_paper_style()
        width = width or PublicationVisualizer.SINGLE_COL_WIDTH
        
        fig, ax = plt.subplots(
            figsize=(width, PublicationVisualizer.COL_HEIGHT)
        )
        
        colors = list(PublicationVisualizer.COLORS.values())
        
        for idx, (model_name, result) in enumerate(results.items()):
            color = colors[idx % len(colors)]
            
            ax.plot(
                result['fpr'],
                result['tpr'],
                color=color,
                lw=2.0,
                label=f"{model_name} (AUC={result['auc']:.3f})"
            )
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PublicationVisualizer.DPI, bbox_inches='tight')
            logger.info(f"Saved ROC curves to {save_path}")
        
        return fig, ax
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
        normalize: bool = False
    ) -> Tuple:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Plot title
            save_path: Path to save figure
            normalize: Normalize by true label count
        
        Returns:
            (figure, axis)
        """
        PublicationVisualizer.set_paper_style()
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        n_classes = cm.shape[0]
        fig, ax = plt.subplots(
            figsize=(PublicationVisualizer.SINGLE_COL_WIDTH + 1, 
                    PublicationVisualizer.COL_HEIGHT + 0.5)
        )
        
        # Plot heatmap
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count' if not normalize else 'Proportion', fontsize=10)
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                value = cm[i, j]
                text = ax.text(
                    j, i, f'{value:{fmt}}',
                    ha='center', va='center',
                    color='white' if value > cm.max() / 2 else 'black',
                    fontsize=9
                )
        
        # Set labels
        if class_names:
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)
        
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PublicationVisualizer.DPI, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        return fig, ax
    
    @staticmethod
    def plot_privacy_utility_tradeoff(
        epsilon_values: List[float],
        accuracy_values: List[float],
        attack_success: List[float],
        model_names: Optional[List[str]] = None,
        title: str = "Privacy-Utility Tradeoff",
        save_path: Optional[str] = None
    ) -> Tuple:
        """Plot privacy-utility tradeoff curves.
        
        Args:
            epsilon_values: DP epsilon values
            accuracy_values: Model accuracy per epsilon
            attack_success: MIA attack success rate per epsilon
            model_names: Names of models
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            (figure, axis)
        """
        PublicationVisualizer.set_paper_style()
        
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(PublicationVisualizer.DOUBLE_COL_WIDTH, 
                    PublicationVisualizer.COL_HEIGHT)
        )
        
        colors = list(PublicationVisualizer.COLORS.values())
        
        # Plot 1: Accuracy vs Epsilon
        for idx in range(len(accuracy_values)):
            color = colors[idx % len(colors)]
            label = model_names[idx] if model_names else f"Model {idx+1}"
            
            ax1.plot(
                epsilon_values,
                accuracy_values[idx],
                'o-',
                color=color,
                lw=2,
                markersize=6,
                label=label
            )
        
        ax1.set_xlabel('Privacy Budget (ε)', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_title('(a) Utility vs Privacy', fontsize=11, fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Attack Success vs Epsilon
        for idx in range(len(attack_success)):
            color = colors[idx % len(colors)]
            label = model_names[idx] if model_names else f"Model {idx+1}"
            
            ax2.plot(
                epsilon_values,
                attack_success[idx],
                's-',
                color=color,
                lw=2,
                markersize=6,
                label=label
            )
        
        ax2.set_xlabel('Privacy Budget (ε)', fontsize=11)
        ax2.set_ylabel('MIA Attack Success (%)', fontsize=11)
        ax2.set_title('(b) Privacy vs Attack Success', fontsize=11, fontweight='bold')
        ax2.set_xscale('log')
        ax2.set_ylim([0, 100])
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PublicationVisualizer.DPI, bbox_inches='tight')
            logger.info(f"Saved privacy-utility tradeoff to {save_path}")
        
        return fig, (ax1, ax2)
    
    @staticmethod
    def plot_communication_rounds(
        round_numbers: np.ndarray,
        accuracy_values: Dict[str, np.ndarray],
        title: str = "Accuracy vs Communication Rounds",
        save_path: Optional[str] = None
    ) -> Tuple:
        """Plot accuracy improvement over federated rounds.
        
        Args:
            round_numbers: Round indices
            accuracy_values: Dict of model_name -> accuracy_per_round
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            (figure, axis)
        """
        PublicationVisualizer.set_paper_style()
        
        fig, ax = plt.subplots(
            figsize=(PublicationVisualizer.SINGLE_COL_WIDTH, 
                    PublicationVisualizer.COL_HEIGHT)
        )
        
        colors = list(PublicationVisualizer.COLORS.values())
        
        for idx, (model_name, accuracies) in enumerate(accuracy_values.items()):
            color = colors[idx % len(colors)]
            
            ax.plot(
                round_numbers,
                accuracies,
                'o-',
                color=color,
                lw=2,
                markersize=5,
                label=model_name
            )
        
        ax.set_xlabel('Communication Round', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PublicationVisualizer.DPI, bbox_inches='tight')
            logger.info(f"Saved communication rounds plot to {save_path}")
        
        return fig, ax
    
    @staticmethod
    def plot_memory_profile(
        model_names: List[str],
        memory_peak: np.ndarray,
        memory_mean: np.ndarray,
        batch_sizes: Optional[List[int]] = None,
        title: str = "Memory Profile",
        save_path: Optional[str] = None
    ) -> Tuple:
        """Plot memory usage comparison.
        
        Args:
            model_names: Model names
            memory_peak: Peak memory per model (MB)
            memory_mean: Mean memory per model (MB)
            batch_sizes: Batch sizes for each measurement
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            (figure, axis)
        """
        PublicationVisualizer.set_paper_style()
        
        fig, ax = plt.subplots(
            figsize=(PublicationVisualizer.SINGLE_COL_WIDTH, 
                    PublicationVisualizer.COL_HEIGHT)
        )
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(
            x - width/2,
            memory_peak,
            width,
            label='Peak Memory',
            color=PublicationVisualizer.COLORS['primary'],
            alpha=0.8
        )
        
        bars2 = ax.bar(
            x + width/2,
            memory_mean,
            width,
            label='Mean Memory',
            color=PublicationVisualizer.COLORS['secondary'],
            alpha=0.8
        )
        
        ax.set_ylabel('Memory (MB)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=8
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PublicationVisualizer.DPI, bbox_inches='tight')
            logger.info(f"Saved memory profile to {save_path}")
        
        return fig, ax
    
    @staticmethod
    def create_results_summary_table(
        results: Dict,
        metrics: List[str],
        model_names: List[str],
        save_path: Optional[str] = None
    ) -> Tuple:
        """Create results summary table.
        
        Args:
            results: Dict with metric values
            metrics: List of metric names
            model_names: List of model names
            save_path: Path to save figure
        
        Returns:
            (figure, axis)
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        
        # Prepare data
        data = []
        for model_name in model_names:
            row = [model_name]
            for metric in metrics:
                key = f"{model_name}_{metric}"
                value = results.get(key, "N/A")
                if isinstance(value, float):
                    row.append(f"{value:.3f}")
                else:
                    row.append(str(value))
            data.append(row)
        
        # Create table
        columns = ['Model'] + metrics
        table = ax.table(
            cellText=data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.2] + [0.15] * len(metrics)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor(PublicationVisualizer.COLORS['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(data) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PublicationVisualizer.DPI, bbox_inches='tight')
            logger.info(f"Saved results table to {save_path}")
        
        return fig, ax
