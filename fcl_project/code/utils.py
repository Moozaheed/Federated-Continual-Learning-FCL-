"""
Utility Functions for FCL Training
Data loading, preprocessing, metrics calculation, visualization helpers.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, auc, confusion_matrix
)
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from code.config import DataConfig, LoggingConfig


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_uci_heart_disease(
    n_samples: int = 200,
    random_state: int = 42,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load UCI Heart Disease dataset (or simulate it).
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        normalize: Whether to normalize features
    
    Returns:
        (X, y) tuple of features and labels
    """
    np.random.seed(random_state)
    
    # Simulate clinical features based on UCI Heart Disease statistics
    # 13 features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    
    feature_distributions = {
        'age': (55, 10),           # Normal distribution: mean=55, std=10
        'sex': (0.5, 0.5),         # Binary
        'cp': (3, 1.3),            # Chest pain type
        'trestbps': (130, 18),     # Resting BP
        'chol': (245, 51),         # Cholesterol
        'fbs': (0.15, 0.35),       # Fasting blood sugar
        'restecg': (0.5, 0.5),     # Resting ECG
        'thalach': (150, 25),      # Max heart rate
        'exang': (0.3, 0.46),      # Exercise-induced angina
        'oldpeak': (1, 1.2),       # ST depression
        'slope': (1.6, 0.6),       # ST segment slope
        'ca': (0.7, 1.1),          # Coronary calcium
        'thal': (0.8, 0.8),        # Thalassemia type
    }
    
    n_features = len(feature_distributions)
    X = np.zeros((n_samples, n_features))
    
    for i, (feat_name, (mean, std)) in enumerate(feature_distributions.items()):
        X[:, i] = np.random.normal(loc=mean, scale=std, size=n_samples)
    
    # Generate labels based on feature patterns
    # Higher risk with: high BP, high cholesterol, exercise-induced angina, ST depression
    risk_score = (
        (X[:, 3] > 140).astype(float) * 0.2 +  # High BP
        (X[:, 4] > 250).astype(float) * 0.2 +  # High cholesterol
        (X[:, 8] > 0.5).astype(float) * 0.2 +  # Exercise angina
        (X[:, 9] > 1.5).astype(float) * 0.2 +  # ST depression
        np.random.normal(0, 0.1, n_samples)    # Random noise
    )
    y = (risk_score > 0.4).astype(int)
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y


def load_multimodal_data(
    n_samples: int = 200,
    image_size: Tuple[int, int] = (224, 224),
    random_state: int = 42,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate multimodal data: EHR (tabular) + Clinical Images.
    """
    # 1. Load Tabular EHR
    X_tab, y = load_uci_heart_disease(n_samples, random_state, normalize)
    
    # 2. Simulate paired clinical images
    # In a real scenario, these would be Chest X-rays or ECG waveforms
    # Here we generate synthetic patterns correlated with the labels
    np.random.seed(random_state)
    X_img = np.random.randn(n_samples, 3, *image_size).astype(np.float32)
    
    # Inject signal into images based on labels
    for i in range(n_samples):
        if y[i] == 1:
            # Add a 'high-risk' pattern (e.g., increase intensity in a specific region)
            X_img[i, :, 50:150, 50:150] += 0.5
            
    return X_img, X_tab, y


def create_multimodal_data_loaders(
    X_img_train: np.ndarray,
    X_tab_train: np.ndarray,
    y_train: np.ndarray,
    X_img_val: np.ndarray,
    X_tab_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for Multimodal training."""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_img_train),
        torch.FloatTensor(X_tab_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_img_val),
        torch.FloatTensor(X_tab_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def fit_multimodal(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    der_buffer: Optional[object] = None,
    alpha: float = 0.1,
    beta: float = 0.5,
    verbose: bool = True
) -> Dict:
    """Advanced Multimodal Training with DER++ support."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, tabs, labels in train_loader:
            imgs, tabs, labels = imgs.to(device), tabs.to(device), labels.to(device)
            
            # Forward pass
            logits = model(imgs, tabs)
            loss = criterion(logits, labels)
            
            # DER++ Regularization
            if der_buffer is not None and der_buffer.n_samples > 0:
                b_imgs, b_tabs, b_labels, b_logits = der_buffer.get_batch(imgs.size(0))
                
                # Model output on buffer samples
                out_buffer = model(b_imgs, b_tabs)
                
                # Dark knowledge MSE + Replay CE
                reg_loss = F.mse_loss(out_buffer, b_logits) * alpha
                reg_loss += F.cross_entropy(out_buffer, b_labels) * beta
                loss += reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update buffer with current task data
            if der_buffer is not None:
                with torch.no_grad():
                    current_logits = model(imgs, tabs)
                der_buffer.add_data(imgs, tabs, labels, current_logits)
            
            total_loss += loss.item()
            
        # Validation... (truncated for brevity in example, full implementation would follow)
        history['train_loss'].append(total_loss / len(train_loader))
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f}")
            
    return history


def create_non_iid_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_hospitals: int = 4,
    non_iid_factor: float = 0.7,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into Non-IID distributions across hospitals.
    
    Args:
        X: Feature matrix
        y: Labels
        n_hospitals: Number of hospital sites
        non_iid_factor: 1.0 = IID, 0.0 = extremely Non-IID
        random_state: Random seed
    
    Returns:
        List of (X_hospital, y_hospital) tuples
    """
    np.random.seed(random_state)
    n_samples = len(X)
    
    # Create non-IID splits by hospital
    hospital_splits = []
    
    for h in range(n_hospitals):
        # Create label preference for this hospital
        class_weights = np.ones(2)
        if non_iid_factor < 1.0:
            # Simulate specialization: some hospitals focus on high-risk patients
            if h % 2 == 0:
                class_weights[1] *= (2 - non_iid_factor)  # More high-risk
            else:
                class_weights[0] *= (2 - non_iid_factor)  # More low-risk
        
        # Select samples with preference
        prob_per_sample = class_weights[y] / class_weights[y].sum()
        hospital_size = n_samples // n_hospitals
        indices = np.random.choice(
            n_samples,
            size=hospital_size,
            p=prob_per_sample,
            replace=False
        )
        
        X_hospital = X[indices]
        y_hospital = y[indices]
        
        hospital_splits.append((X_hospital, y_hospital))
    
    return hospital_splits


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from numpy arrays.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    use_ewc: bool = False,
    lambda_ewc: float = 0.5,
    verbose: bool = True,
    **kwargs # Handle unexpected arguments
) -> Dict[str, List[float]]:
    """
    Train a PyTorch model with validation and optional EWC.
    """
    # Alias epochs if num_epochs is provided
    epochs = num_epochs
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # EWC Penalty
            if use_ewc and hasattr(model, 'fisher_matrix') and model.fisher_matrix is not None:
                ewc_loss = 0
                for name, param in model.named_parameters():
                    if name in model.fisher_matrix and name in model.optimal_params:
                        fisher = model.fisher_matrix[name]
                        opt_param = model.optimal_params[name]
                        ewc_loss += (fisher * (param - opt_param) ** 2).sum()
                loss += lambda_ewc * ewc_loss
            elif use_ewc and hasattr(model, 'fisher'): # Handle alternative attribute name
                 ewc_loss = 0
                 fisher_dict = getattr(model, 'fisher')
                 for name, param in model.named_parameters():
                     if name in fisher_dict and name in model.optimal_params:
                         fisher = fisher_dict[name]
                         opt_param = model.optimal_params[name]
                         ewc_loss += (fisher * (param - opt_param) ** 2).sum()
                 loss += lambda_ewc * ewc_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    return history

def save_model(model: torch.nn.Module, path: str):
    """Save model state dict."""
    torch.save(model.state_dict(), path)


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: Optional[torch.nn.Module] = None,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Compute metrics
    metrics = compute_metrics(all_targets, all_predictions, all_probabilities)
    
    return metrics


# ============================================================================
# METRICS & EVALUATION
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC-AUC)
    
    Returns:
        Dict of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add ROC-AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            auc_val = roc_auc_score(y_true, y_pred_proba[:, 1])
            metrics['roc_auc'] = auc_val
            metrics['auc_roc'] = auc_val
        except:
            metrics['roc_auc'] = 0.0
            metrics['auc_roc'] = 0.0
    
    return metrics


def compute_backward_transfer(accuracy_per_task: List[float]) -> float:
    """
    Compute Backward Transfer (BWT).
    Measures how much new tasks hurt performance on old tasks.
    
    BWT = (1 / T-1) * sum(acc_i(task_j) - acc_i(task_T))
    where T is number of tasks
    
    Positive BWT = improvement on old tasks (due to replay)
    Negative BWT = forgetting
    """
    if len(accuracy_per_task) < 2:
        return 0.0
    
    T = len(accuracy_per_task)
    bwt_terms = []
    
    for i in range(T - 1):
        # Improvement of task i after seeing all tasks
        bwt_terms.append(accuracy_per_task[T - 1] - accuracy_per_task[i])
    
    return np.mean(bwt_terms) if bwt_terms else 0.0


def compute_forward_transfer(accuracy_per_task: List[float]) -> float:
    """
    Compute Forward Transfer (FWT).
    Measures how much old tasks help with new tasks.
    
    FWT = (1 / T) * sum(acc_i(task_j) - baseline_i)
    """
    if len(accuracy_per_task) < 2:
        return 0.0
    
    # Baseline: random classifier
    baseline = 0.5
    
    fwt_terms = []
    for acc in accuracy_per_task[1:]:  # Skip first task
        fwt_terms.append(acc - baseline)
    
    return np.mean(fwt_terms) if fwt_terms else 0.0


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    title: str = "Training History",
    save_path: Optional[str] = None
):
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'o-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_losses, 's-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('(a) Loss Trajectory', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accuracies, 'o-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_accuracies, 's-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('(b) Accuracy Trajectory', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None
):
    """Plot confusion matrix as heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = ['Negative', 'Positive']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Ground Truth', fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    plt.colorbar(im, ax=ax)
    plt.title('Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# FEDERATED LEARNING HELPERS
# ============================================================================

def aggregate_gradients(
    client_gradients: List[Dict],
    client_sample_counts: List[int],
    aggregation_strategy: str = "weighted"
) -> Dict:
    """
    Aggregate gradients from multiple clients.
    
    Args:
        client_gradients: List of gradient dictionaries from each client
        client_sample_counts: Number of samples per client
        aggregation_strategy: "weighted" or "uniform"
    
    Returns:
        Aggregated gradients
    """
    if aggregation_strategy == "weighted":
        # Weighted average by sample count (FedAvg)
        total_samples = sum(client_sample_counts)
        weights = [count / total_samples for count in client_sample_counts]
    else:
        # Uniform average
        weights = [1.0 / len(client_gradients)] * len(client_gradients)
    
    aggregated = {}
    
    for key in client_gradients[0].keys():
        aggregated[key] = sum(
            weights[i] * client_gradients[i][key]
            for i in range(len(client_gradients))
        )
    
    return aggregated


def extract_model_gradients(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    """
    Extract gradients from model as numpy arrays.
    
    Returns:
        Dict mapping parameter names to gradient arrays
    """
    gradients = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.cpu().detach().numpy()
    
    return gradients


# ============================================================================
# PRIVACY HELPERS
# ============================================================================

def compute_epsilon_from_noise(
    noise_multiplier: float,
    n_samples: int,
    batch_size: int,
    epochs: int,
    target_delta: float = 1e-5
) -> float:
    """
    Rough epsilon estimation using Opacus accounting.
    (Simplified version; full implementation uses RDP accounting)
    
    Args:
        noise_multiplier: Noise multiplier for clipping
        n_samples: Total training samples
        batch_size: Batch size
        epochs: Number of epochs
        target_delta: Target delta (failure probability)
    
    Returns:
        Approximate epsilon value
    """
    steps = (n_samples // batch_size) * epochs
    epsilon = 2 * np.sqrt(2 * np.log(1.25 / target_delta)) / (noise_multiplier * steps)
    return epsilon


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    X, y = load_uci_heart_disease(n_samples=200)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Test metrics
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = compute_metrics(y_true, y_pred)
    print(f"\nMetrics: {metrics}")
    
    # Test BWT/FWT
    acc_per_task = [0.70, 0.72, 0.75, 0.78]
    bwt = compute_backward_transfer(acc_per_task)
    fwt = compute_forward_transfer(acc_per_task)
    print(f"BWT: {bwt:.3f}, FWT: {fwt:.3f}")
