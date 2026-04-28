# Long-term Planning: IEEE-Targeted FCL Research

## Phase 1: Prototype Development (Weeks 1-3)
*   **Data:** Use UCI Heart Disease (4 locations) for rapid prototyping.
*   **Infrastructure:** Setup Flower (flwr) with a custom `Client` class capable of handling local VAE training.
*   **Baseline:** Document the Catastrophic Forgetting when switching from Cleveland to Switzerland.

## Phase 2: The Generative Innovation (Weeks 4-7)
*   **Task:** Implement a **Variational Autoencoder (VAE)** at each client.
*   **Integration:** Develop the **Generative Replay** logic: When the model moves to a "New Hospital," the "Old Hospital's" VAE generates synthetic patients to mix into the training batch.
*   **Privacy:** Integrate **Differential Privacy (DP)** using the `Opacus` library (PyTorch) to ensure VAE-generated data doesn't leak raw patient info.

## Phase 3: Scaling to Clinical Grade (Weeks 8-10)
*   **Data Migration:** Transition to **MIMIC-IV** (Structured EHR data).
*   **Simulation:** Create a temporal split (e.g., 2012-2015 data vs 2016-2019 data) to simulate real-world Concept Drift in an ICU setting.
*   **Validation:** Run the VAE-based FCL on MIMIC-IV and compare against standard FedAvg and EWC.

## Phase 4: Comparative Analysis (Weeks 11-12)
*   Compare results against 2024-2025 SOTA benchmarks.
*   Measure the **Privacy-Utility Trade-off** (how much accuracy do we lose when we increase Differential Privacy noise?).

## Phase 5: Publication Writing (Weeks 13-16)
*   Draft the IEEE paper using the LaTeX template.
*   Finalize the GitHub repository with a "One-Click Reproduce" script.
