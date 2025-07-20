# Code Refactoring Assistant - A Proof-of-Concept

This repository outlines a proof-of-concept for a lightweight, open-source system for code refactoring. It demonstrates how a robust MLOps pipeline for model training can be combined with a simple desktop application and a VS Code plugin for local inference. The goal is to showcase a complete conceptual workflow, from training a specialized model to using it for a development task.

## Key Components & Directory Structure

The repository is organized into four main functional areas:

- **`app/`**: Contains a demo desktop application (built with PyQt5) and an integrated FastAPI server. Its purpose is to provide a simple GUI for managing the local inference server and downloading models for demonstration.

- **`airflow/`**: Holds the Apache Airflow configuration, including DAGs for orchestrating the conceptual MLOps training and evaluation pipelines.

- **`mlflow/`**: Contains the core Jupyter notebooks for model fine-tuning (`train.ipynb`) and evaluation (`evaluate.ipynb`). These are designed to be executed as jobs by Databricks to showcase the training part of the concept.

- **`plugin/`**: Contains the source code for a demo Visual Studio Code extension. This plugin serves as a basic client that communicates with the local FastAPI server to showcase how refactoring suggestions could be provided directly within an editor.

## System Dataflow

The following dataflow illustrates the conceptual design of the system, which is divided into two distinct flows.

### Flow 1: Model Training and Registration (MLOps Pipeline Concept)

This flow is centered around the MLOps automation tools and demonstrates how to create and validate new model versions.

1.  An **Airflow DAG** is manually triggered, initiating the pipeline.
2.  The DAG executes jobs on **Databricks**, running the code from the Jupyter Notebooks in the `mlflow/` directory.
3.  The `train.ipynb` notebook fine-tunes a language model and logs the resulting artifacts to the **MLflow Model Registry**.
4.  Next, `evaluate.ipynb` runs benchmarks on the new model and attaches the evaluation metrics as tags to that version in the registry.

### Flow 2: Demo of Local Inference Workflow

This flow demonstrates how a developer could leverage the trained models for code refactoring on their local machine.

1.  The user first launches the **Demo Desktop Application** (`app/main.py`) to manage models. It connects to the MLflow Model Registry, allowing a user to download a model and start the local API server.
2.  The `ApiServerManager` utility launches a **FastAPI server** process (`app/api/`), which loads the selected model and exposes a `/refactor` endpoint on `localhost`.
3.  The **Demo VS Code Extension** (from `plugin/`) communicates with this local API. When a user requests a refactoring, the plugin sends the code to the `/refactor` endpoint.
4.  The FastAPI server processes the code using the loaded model and returns the refactored version.
5.  The VS Code extension receives the response and displays the AI-generated refactoring suggestion directly to the user in the editor, completing the proof-of-concept workflow. 