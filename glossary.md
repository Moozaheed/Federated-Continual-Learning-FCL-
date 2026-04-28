# Master Keyword Glossary: Federated Continual Learning (FCL)

## 1. The Federated Learning (FL) Domain
*   **Aggregator (Server):** The central entity that collects model updates (not raw data) from clients and combines them.
*   **Clients (Nodes):** The local entities (e.g., Hospital A, Clinic B) that train the model on their own private data.
*   **Communication Round:** One full cycle where the server sends a model, clients train it locally, and send updates back.
*   **FedAvg (Federated Averaging):** The most common algorithm; it simply averages the weights of all client models.
*   **Non-IID (Independent and Identically Distributed):** The challenge where Hospital A's data distribution differs from Hospital B's (e.g., different demographics).
*   **Secure Aggregation:** A cryptographic technique ensuring the server sees only the *sum* of updates, not individual client contributions.

## 2. The Continual Learning (CL) Domain
*   **Catastrophic Forgetting:** When a model loses old knowledge (Task A) upon learning new knowledge (Task B).
*   **Stability-Plasticity Dilemma:** The trade-off between keeping old knowledge (stability) and learning new information (plasticity).
*   **Elastic Weight Consolidation (EWC):** A regularization technique that penalizes changes to weights important for previous tasks.
*   **Experience Replay (Buffer):** Storing and revisiting a small subset of old data to prevent forgetting.
*   **Generative Replay:** Using generative models (GANs/VAEs) to create synthetic "old" data for rehearsal, preserving privacy.
*   **Knowledge Distillation:** Using a previous version of the model to guide the current version in maintaining old performance.

## 3. The Concept Drift Domain
*   **Concept Drift:** A change in the statistical properties of the target variable over time.
*   **Covariate Shift (Feature Drift):** Changes in the distribution of input features (e.g., different lab equipment).
*   **Label Shift:** Changes in the distribution of the target classes (e.g., a sudden outbreak).
*   **Abrupt vs. Gradual Drift:** Sudden changes (new hospital) vs. slow changes (aging population).
*   **Drift Detection:** Monitoring error rates (e.g., using **ADWIN**) to identify when data patterns have changed.

## 4. The Privacy & Security Domain
*   **Differential Privacy (DP):** Adding noise to updates to prevent the identification of specific individuals in the training set.
*   **Membership Inference Attack:** An attempt to determine if a specific record was part of the training data.
*   **Trusted Execution Environment (TEE):** Secure hardware regions for performing sensitive computations.

## 5. Research Evaluation Metrics
*   **Backward Transfer (BWT):** Measure of how much new learning influences performance on previous tasks.
*   **Forward Transfer (FWT):** Measure of how much previous learning helps in acquiring new tasks.
*   **Average Accuracy:** Overall performance across all tasks/clients.
*   **Communication Overhead:** The total data transferred between clients and the server.
