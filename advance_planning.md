# Advance Planning: Powerful Multimodal FCL Run

This document outlines the architectural and technical roadmap for a "Powerful Run" of the Federated Continual Learning (FCL) framework, optimized for an **RTX 3075 (8GB VRAM)** and **32GB RAM**.

---

## 1. Hardware Utilization Strategy

### GPU (RTX 3075 - 8GB VRAM)
- **Primary Use**: Processing deep feature extractions from images (MobileNet branch) and attention mechanisms (FT-Transformer branch).
- **Optimizations**: 
  - Increase batch size to **128-256** to maximize parallel throughput.
  - Enable `torch.backends.cudnn.benchmark = True`.
  - Use `Automatic Mixed Precision (AMP)` to speed up training and save VRAM.

### RAM (32GB)
- **Primary Use**: Storing large-scale synthetic datasets and advanced replay buffers (DER++).
- **Scale**: Generate 100,000+ paired (Image + Tabular) clinical samples directly in system memory to eliminate I/O bottlenecks.

---

## 2. Multimodal Architecture

We will implement a dual-branch fusion network designed to replicate modern clinical decision support systems.

### Branch A: Image Extractor (MobileNetV3)
- **Architecture**: `MobileNetV3-Small` (or Large) pre-trained on ImageNet.
- **Role**: Process 3-channel clinical images (e.g., Chest X-rays, MRI slices).
- **Modification**: Replace the classification head with a Global Average Pooling layer to output a compact feature vector (e.g., 576-dim).

### Branch B: EHR Extractor (Enhanced FT-Transformer)
- **Architecture**: Scaled-up version of the current model.
- **Role**: Process tabular Electronic Health Record (EHR) data.
- **Changes**: Increase `token_dim` to 192 and `n_transformer_blocks` to 6.

### Fusion & Prediction Head
- **Mechanism**: Feature concatenation followed by a multi-layer MLP fusion network.
- **Output**: Multi-task or multi-label clinical predictions.

---

## 3. Advanced Continual Learning: DER++

To eliminate catastrophic forgetting across hospital sites, we will transition from EWC to **Dark Experience Replay (DER++)**.

- **Mechanism**: Saves a memory-efficient replay buffer of past samples and their associated "logits" (model output probabilities).
- **Benefit**: Retains the "dark knowledge" (nuanced output distributions) from previous tasks, providing much better stability than standard replay or regularization.

---

## 4. Required Modifications (Roadmap)

To implement the above, the following adjustments will be needed in the future:

### `code/config.py`
- [ ] **Add `MultimodalConfig`**: Define image resolution (224x224), channels, and CNN parameters.
- [ ] **Adjust `ModelConfig`**: Increase default embedding sizes and block counts.
- [ ] **Add `DERConfig`**: Define buffer size (e.g., 2,000 samples) and regularization weights ($\alpha, \beta$).

### `code/model.py`
- [ ] **Implement `MobileNetExtractor`**: Wrapper class for `torchvision` models.
- [ ] **Implement `MultimodalFCLModel`**: Fusion class combining CNN and Transformer branches.
- [ ] **Add DER++ support**: Methods to handle logit-based training.

### `code/utils.py`
- [ ] **Add Multimodal Data Generator**: Create paired image tensors and tabular arrays.
- [ ] **Update `fit` & `evaluate`**: Modify signatures to accept `(images, tabular)` inputs.
- [ ] **Implement Replay Buffer**: Logic for sampling and updating the memory buffer.

---

## 5. Advanced Datasets & Data Strategy

To achieve state-of-the-art results, we will transition from synthetic data to the industry's most rigorous clinical datasets.

### Primary Target: MIMIC Multimodal Ecosystem
- **MIMIC-IV (Tabular EHR)**: 
  - **Scope**: Comprehensive records for ~40,000 Intensive Care Unit (ICU) patients.
  - **Features**: Laboratory measurements, vital signs, medications, and hourly physiological data (100+ variables).
- **MIMIC-CXR (Clinical Images)**: 
  - **Scope**: 377,110 Chest X-ray images.
  - **Multimodal Link**: Patients are linked via `subject_id`, allowing the model to fuse X-ray image features with longitudinal EHR lab data for high-accuracy diagnosis.

### Fallback/Immediate Alternative: MedMNIST v2
- **Scope**: A large-scale benchmark of 12 standardized biomedical datasets (e.g., BloodMNIST, PathMNIST, ChestMNIST).
- **Use Case**: Ideal for rapid prototyping of **Federated Continual Learning** tasks because it provides standardized image sizes (28x28 or 224x224) and clear task boundaries.

### Data Processing Strategy
1. **Feature Engineering**: Implement a pipeline to extract "Temporal EHR windows" (e.g., the last 24 hours of vitals) before an image was taken.
2. **Privacy-Preserving Preprocessing**: Standardize all images to 224x224 and normalize tabular features per hospital site to simulate real-world federated constraints.

---

## 6. Deliverables

- **New Notebook**: `FCL_Power_Run_Multimodal.ipynb`.
- **Expanded Infrastructure**: A framework capable of state-of-the-art multimodal research.

---

**Status**: Planning Phase | **Target**: SOTA Healthcare FCL Implementation
