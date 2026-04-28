# Model Design Plan: FT-Transformer (Feature Tokenizer Transformer)

This file outlines the technical architecture for `model.py` aimed at an IEEE-level publication.

## 1. Core Architecture: FT-Transformer
Instead of a flat Multi-Layer Perceptron (MLP), we will implement an **FT-Transformer**. 

### A. Feature Tokenizer Layer
*   **Numerical Features:** Each feature (e.g., Blood Pressure) is multiplied by a learnable weight vector to project it into an $d$-dimensional embedding space.
*   **Categorical Features:** Standard embedding layers to convert categories (e.g., Gender) into $d$-dimensional vectors.
*   **Output:** A sequence of "tokens" (one per feature), similar to words in a sentence.

### B. Transformer Encoder Blocks
*   **Multi-Head Self-Attention (MHSA):** Allows the model to learn which features are most important *in relation to others* (e.g., how Age interacts with Cholesterol).
*   **Layer Normalization & Residual Connections:** For training stability.
*   **Feed-Forward Network (FFN):** Position-wise dense layers.

### C. Prediction Head
*   A classification layer that takes the [CLS] token or a global average pool of the features to output the heart disease probability.

## 2. Advanced Continual Learning Features (Innovation)
To solve **Concept Drift**, we will include:
*   **Learnable Prompts (Prompt-Tuning):** We will add extra "prefix tokens" to the input. When a new hospital joins, the core Transformer weights are **frozen**, and only these prompts are trained. This prevents Catastrophic Forgetting.
*   **EWC Compatibility:** Weights will have a hook to calculate Fisher Information Matrices for regularization.

## 3. Federated Learning Features
*   **Proximal Term Support:** A method to calculate the L2 distance between local weights and global weights to support the **FedProx** algorithm.
*   **Opacus/DP Support:** The model will be designed with a compatible batch-norm (or LayerNorm) to ensure it works with **Differential Privacy** gradient clipping.

## 4. Why this is "High Impact"
1.  **Tabular SOTA:** Transformers are currently the best performing models for tabular data in research.
2.  **Scalability:** Prompts are much smaller than model weights (KB vs MB), making the federated communication extremely efficient.
3.  **Stability:** Transformers are more robust to the "Noisy" updates common in medical federated settings.
