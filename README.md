# Lightweight Code Refactoring Assistant

This repository contains the core components for a lightweight, open-source system for code refactoring. It implements a full end-to-end MLOps pipeline for fine-tuning a specialized language model and a local application stack for developers to use the model for inference. The system is built around the `deepseek-coder-1.3b-instruct` model, fine-tuned specifically for Python code enhancement tasks.

## System Architecture

The project is divided into two primary, interconnected flows: a cloud-based MLOps pipeline for model production and a local deployment solution for developer use.

### 1. MLOps Pipeline: Model Training & Deployment

This flow automates the creation, evaluation, and versioning of new models.

1.  **Orchestration**: An **Apache Airflow DAG** (`train_and_evaluate_dag.py`) is manually triggered to begin the pipeline. It fetches the raw `CodeSearchNet` dataset from Azure Blob Storage and prepares it for training.
2.  **Two-Stage Training**: The DAG executes jobs on **Databricks**, running two sequential notebooks:
    *   `pretrain.ipynb`: Performs "Continued Pre-training" on raw Python code to enhance the model's foundational understanding.
    *   `train.ipynb`: Performs "Instruction Fine-Tuning" using function-docstring pairs to teach the model the refactoring task.
3.  **Evaluation**: An `evaluate.ipynb` notebook benchmarks the newly trained "challenger" model against the current "champion" from production using the **BLEURT** metric on a held-out test set (`tests.yaml`).
4.  **Versioning & Deployment**: The models, parameters, and metrics are tracked in **MLflow**. If the challenger scores higher than the champion, it is automatically promoted to the `Staging` stage in the MLflow Model Registry, awaiting manual approval for `Production`.

### 2. Local Inference: Desktop Application & VS Code Extension

This flow enables developers to use the production-ready models on their local machines securely and efficiently.

1.  **Desktop Application**: The user launches a standalone desktop application (`app/`) to manage models. It connects to the MLflow Model Registry to download the latest `Production` version of the model.
2.  **Local API Server**: The application starts a local **FastAPI** server (`app/api/`) in the background. This server loads the downloaded model and exposes a `/refactor` endpoint.
3.  **VS Code Extension**: A companion **VS Code extension** (`plugin/`) communicates with the local API. When a developer selects a block of code and requests a refactoring, the extension sends the code to the local server and displays the returned suggestion directly in the editor.

## Repository Structure

The key components within this `repo/` directory are organized as follows:

-   **`app/`**: The local deployment solution, containing the desktop GUI (PyQt5) for model management and the FastAPI serving layer.
-   **`airflow/`**: Contains Airflow configurations and the primary DAG (`train_and_evaluate_dag.py`) for orchestrating the MLOps pipeline.
-   **`mlflow/`**: Contains the core Databricks notebooks for the two-stage model training (`pretrain.ipynb`, `train.ipynb`) and evaluation (`evaluate.ipynb`).
-   **`plugin/`**: Source code for the Visual Studio Code extension that acts as a client to the local inference server.

## Key Technologies

-   **Model**: `deepseek-coder-1.3b-instruct`
-   **Training Techniques**: Parameter-Efficient Fine-Tuning (PEFT), Low-Rank Adaptation (LoRA)
-   **Dataset**: `CodeSearchNet` (Python subset)
-   **MLOps & Infrastructure**: Airflow, Databricks, MLflow, Azure Blob Storage
-   **Local Application**: Python, FastAPI, PyQt5
-   **Client**: Visual Studio Code Extension 