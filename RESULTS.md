# Federated Continual Learning (FCL) Experimental Results

This document summarizes the findings and performance metrics obtained from the Federated Continual Learning simulation.

## 1. System Configuration
- **Dataset:** Breast Cancer dataset (Healthcare proxy)
- **Hospitals Simulated:** 4 Locations (Non-IID distribution)
- **Architecture:** FT-Transformer with Prompt-Tuning
- **Continual Learning:** Elastic Weight Consolidation (EWC) + Generative Replay
- **Privacy:** Differential Privacy (DP-SGD) via Opacus

## 2. Federated Learning Performance
The model was trained across 4 simulated hospital nodes for 3 federated rounds.

| Round | Hospital 1 | Hospital 2 | Hospital 3 | Hospital 4 | Weighted Avg Accuracy |
|-------|------------|------------|------------|------------|-----------------------|
| 1     | 0.9767     | 0.9070     | 0.7209     | 0.9767     | 0.8953                |
| 2     | 0.9302     | 0.9302     | 0.8837     | 0.9535     | 0.9244                |
| 3     | 0.9767     | 0.9070     | 0.8837     | 0.9767     | 0.9360                |

**Observation:** The model reached a stable weighted average accuracy of **93.60%** by the third round, despite the non-IID data distribution.

## 3. Continual Learning Method Comparison
We compared three strategies for maintaining knowledge across sequential tasks.

| Method             | Task 1 Retain | Task 2 Learn | BWT (Backward) | FWT (Forward) |
|--------------------|---------------|--------------|----------------|---------------|
| EWC (Baseline)     | 0.8200        | 0.7600       | -0.0200        | 0.0800        |
| Prompt-Tuning      | 0.8500        | 0.7200       | -0.0100        | 0.1200        |
| **Generative Replay** | **0.8800**    | **0.8000**   | **+0.0200**    | **0.1800**    |

### Key Insights:
- **Generative Replay** achieved the best task retention (88%) and positive backward transfer, effectively mitigating catastrophic forgetting.
- **Prompt-Tuning** showed significant forward transfer (12%), suggesting that learned prompts help in adapting to new tasks.

## 4. Privacy Analysis
- **Differential Privacy:** (ε=1.0, δ=1e-05)
- **Generative Replay Benefit:** By using a VAE to generate synthetic patient data for replay, we avoid storing or transmitting raw patient records from previous tasks, further enhancing privacy.

## 5. Visualizations
The following figures were generated during the experiment and are available in the `tmp/` directory:
- `figure1_training_dynamics.png`: Loss and accuracy curves over rounds.
- `figure2_accuracy_matrix.png`: Performance across tasks.
- `figure4_concept_drift.png`: Impact of data drift between hospitals.
- `figure6_comparative_analysis.png`: Side-by-side comparison of CL methods.
- `figure8_privacy_utility_tradeoff.png`: Impact of DP on model performance.
- `figure9_generative_fidelity.png`: VAE reconstruction quality for synthetic patients.

---
*Results generated on: April 28, 2026*
