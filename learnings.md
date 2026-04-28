# Learning Roadmap: Federated Continual Learning (FCL) & Concept Drift

Focus: Solving the "Catastrophic Forgetting" and "Data Shift" problems in distributed healthcare systems.

## 1. Fundamentals (Prerequisites)
*   **Deep Learning Review:**
    *   Optimization (SGD, Adam, Weight Decay).
    *   Architectures: CNNs (Medical Imaging) and LSTMs/Transformers (Time-series EHR data).
*   **Medical Data Handling:**
    *   Preprocessing DICOM images.
    *   Handling missing values in Electronic Health Records (EHR).
    *   Understanding HIPAA/GDPR constraints on data sharing.

## 2. Federated Learning (FL) Deep Dive
*   **Architectures:** Centralized (Client-Server) vs. Decentralized (Peer-to-Peer).
*   **Algorithms:**
    *   **FedAvg:** The baseline.
    *   **FedProx:** Handling heterogeneity in client hardware.
    *   **FedOpt:** Adaptive optimizers in FL.
*   **Challenges:**
    *   **Non-IID Data:** Why hospital A's data looks different from hospital B's.
    *   **Communication Efficiency:** Reducing the size of model updates.

## 3. Continual Learning (CL) Foundations
*   **The Stability-Plasticity Dilemma:** Balancing new knowledge vs. old knowledge.
*   **Catastrophic Forgetting:** Why models "break" when they see new data distributions.
*   **Methodologies:**
    *   **Regularization-based:** Elastic Weight Consolidation (EWC), MAS.
    *   **Rehearsal/Replay:** Generative Replay vs. Experience Replay (Buffer management).
    *   **Architecture-based:** Progressive Neural Networks, Parameter Isolation.

## 4. Concept Drift & Non-Stationarity
*   **Types of Drift:**
    *   **Abrupt:** Sudden changes (e.g., a new pandemic).
    *   **Gradual:** Slow changes (e.g., aging population).
    *   **Incremental:** Evolution of medical equipment over years.
*   **Detection Algorithms:** ADWIN (Adaptive Windowing), DDM (Drift Detection Method).
*   **Healthcare Context:** Seasonal disease drift, clinical protocol changes.

## 5. The Frontier: Federated Continual Learning (FCL)
*   **Dual-Level Forgetting:**
    *   **Local Forgetting:** A hospital forgets its own past patients.
    *   **Global Forgetting:** The central server forgets knowledge from hospitals it hasn't heard from recently.
*   **Global Memory Management:** How to maintain a "global replay buffer" without violating privacy (Differential Privacy + Generative Models).
*   **Asynchronous FCL:** Handling clients that join/leave the training process at different times.

## 6. Frameworks to Master
*   **Flower (flwr.dev):** Best for production-grade FL.
*   **Avalanche (ContinualAI):** The industry standard for Continual Learning.
*   **PySyft:** For privacy-preserving operations.
