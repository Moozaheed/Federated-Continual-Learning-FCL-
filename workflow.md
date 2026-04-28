# Local Development Workflow: "Develop Locally, Scale Globally"

To ensure impact and speed, you will use your local machine as the **Command Center** and the Cloud as the **Execution Engine**.

## 1. Local Environment Setup
*   **IDE:** VS Code is recommended.
*   **Virtual Environment:** Use `conda` or `venv`.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install torch flwr pandas numpy scikit-learn opacus wandb
    ```
*   **Data Handling:** Keep a "Mini-Batch" of the UCI data locally for 1-minute debugging loops. Don't try to run a full 100-epoch simulation locally.

## 2. The Development Cycle
1.  **Code (Local):** Write the model, server, and client logic.
2.  **Unit Test (Local):** Run a single "Communication Round" with 2 virtual clients on your PC to ensure no syntax errors.
3.  **Containerize (Docker):** Package the code into a Docker image.
4.  **Push (Artifact Registry):** Push the image to Google Cloud.
5.  **Execute (GKE):** Spin up the Kubernetes cluster to run the full research experiment.

## 3. Remote Synchronization
*   **Git:** All changes must be pushed to a private GitHub/GitLab repo.
*   **Weights & Biases (W&B):** While the code runs in the cloud, you will watch the live accuracy graphs on your local browser via [wandb.ai](https://wandb.ai).

## 4. Local Tools for Research
*   **Jupyter Notebooks:** Use locally for Exploratory Data Analysis (EDA) and final plot generation.
*   **Postman:** If you build any APIs for your dashboard later.
