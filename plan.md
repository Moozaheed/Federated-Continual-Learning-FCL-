# Research Plan: Federated Continual Learning (FCL) for Concept Drift in Healthcare

## Phase 1: Preparation & Environment (Weeks 1-2)
*   **Goal:** Setup a reproducible simulation environment.
*   **Tasks:**
    1.  Install Python 3.10+, PyTorch, Flower (FL), and Avalanche (CL).
    2.  Select a Dataset: 
        *   *Option A:* **MIMIC-IV** (Complex, real-world EHR).
        *   *Option B:* **UCI Heart Disease** or **Diabetes** (Simpler, good for proof of concept).
    3.  Create a "Data Splitter" script that simulates different hospitals (Clients) with different data distributions (Non-IID).

## Phase 2: Building the Baseline FL (Weeks 3-4)
*   **Goal:** Establish a standard Federated Learning benchmark.
*   **Tasks:**
    1.  Implement a standard `FedAvg` server and client using the Flower framework.
    2.  Train the model across 5 virtual hospitals.
    3.  Evaluate on a global test set.
    4.  **Measurement:** Record "Global Accuracy" and "Communication Overhead."

## Phase 3: Introducing Concept Drift (Weeks 5-6)
*   **Goal:** Mathematically simulate "real-world" changes in healthcare.
*   **Tasks:**
    1.  Introduce **Feature Drift:** Suddenly change the scale or distribution of a vital sign (e.g., Blood Pressure) in one hospital.
    2.  Introduce **Label Drift:** Change the prevalence of a disease over "time" in the simulation.
    3.  **Observation:** Show how the standard `FedAvg` model accuracy drops when it encounters this drift.

## Phase 4: Implementing FCL Strategies (Weeks 7-12)
*   **Goal:** Apply Continual Learning methods to the Federated setup.
*   **Tasks:**
    1.  **Level 1 (Regularization):** Integrate EWC (Elastic Weight Consolidation) into the local training step.
    2.  **Level 2 (Replay):** Implement a "Privacy-Preserving Replay Buffer." Since we can't share raw data, use a **Generative Adversarial Network (GAN)** or **VAE** at each hospital to generate synthetic "old data" for rehearsal.
    3.  **Level 3 (Aggregation):** Modify the server-side aggregation to weigh "updated" models differently if they have adapted to drift.

## Phase 5: Evaluation & Validation (Weeks 13-16)
*   **Goal:** Prove the impact of your research.
*   **Metrics to Track:**
    *   **Average Accuracy:** Overall performance.
    *   **Backward Transfer (BWT):** How much does learning from Hospital C hurt performance on Hospital A's old data? (Measures forgetting).
    *   **Forward Transfer (FWT):** Does learning from old hospitals help the model learn faster at a new hospital?
    *   **Privacy Budget:** Using Differential Privacy metrics to ensure the "Replay" doesn't leak patient identity.

## Phase 6: Impact & Dissemination (Weeks 17-20)
*   **Goal:** Share your findings.
*   **Tasks:**
    1.  Write a technical blog post or a research paper draft.
    2.  Clean up the code for a GitHub repository (essential for SE career impact).
    3.  Create a "Dashboard" visualizing how the model adapts to drift in real-time.
