import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add path
fcl_project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, fcl_project_path)

from code.config import config
from code.multimodal.fusion import MultimodalFCLModel
from code.der import DERBuffer
from code.utils import load_multimodal_data, create_multimodal_data_loaders, fit_multimodal

print("🚀 Starting Advanced Multimodal FCL Power Run")

# 1. Setup Device & Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. Initialize Multimodal Model
model = MultimodalFCLModel(config).to(device)
print(f"Multimodal Model Initialized. Params: {sum(p.numel() for p in model.parameters()):,}")

# 3. Initialize DER++ Buffer
buffer = DERBuffer(buffer_size=config.der.buffer_size, device=device)
print(f"DER++ Replay Buffer ready (Size: {config.der.buffer_size})")

# 4. Simulated Continual Learning Tasks (Multimodal)
n_tasks = 3
results = []

for task_id in range(1, n_tasks + 1):
    print(f"\n--- TASK {task_id}/{n_tasks} ---")
    
    # Generate Multimodal Task Data
    X_img, X_tab, y = load_multimodal_data(n_samples=500, random_state=42+task_id)
    
    # Split
    split = int(0.8 * len(y))
    train_loader, val_loader = create_multimodal_data_loaders(
        X_img[:split], X_tab[:split], y[:split],
        X_img[split:], X_tab[split:], y[split:],
        batch_size=config.training.batch_size
    )
    
    # Train with DER++
    history = fit_multimodal(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5, # Reduced for power run demo
        learning_rate=config.training.learning_rate,
        device=device,
        der_buffer=buffer,
        alpha=config.der.alpha,
        beta=config.der.beta
    )
    
    print(f"Task {task_id} Completed. Final Train Loss: {history['train_loss'][-1]:.4f}")

print("\n✅ Power Run Successful")
