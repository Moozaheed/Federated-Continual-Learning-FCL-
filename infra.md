# Cloud Infrastructure & MLOps Strategy

## 1. Compute & Orchestration
*   **Platform:** Google Cloud Platform (GCP).
*   **Environment:** Google Kubernetes Engine (GKE).
*   **Logic:** Each Federated Client (Hospital) is a containerized Pod to ensure true isolation.
*   **Scaling:** Use Cluster Autoscaler to simulate up to 100+ hospitals (crucial for "Impact" in paper).

## 2. Technical Stack
*   **Core:** PyTorch + Flower (flwr.dev).
*   **Privacy:** Opacus (Differential Privacy library).
*   **Experiment Tracking:** Weights & Biases (W&B).
*   **Data Versioning:** DVC (Data Version Control).

## 3. Security & Compliance (Simulation)
*   **Encryption:** GCP KMS for data-at-rest.
*   **Secrets:** GCP Secret Manager for API keys and W&B tokens.
*   **Isolation:** VPC Service Controls to prevent data exfiltration between hospital pods.

## 4. Cost Optimization
*   **Spot Instances:** Use Preemptible N1/N2 instances with T4 GPUs.
*   **Budgeting:** Set up GCP billing alerts at $10, $50, and $100 intervals.

## 5. Industry Best Practices
*   **Reproducibility:** All environments defined via `Dockerfile` and `Terraform`.
*   **Logging:** Centralized logs via Cloud Logging (Stackdriver) to debug client failures in real-time.
