from __future__ import annotations

import base64
import json
import os
import pickle
import tempfile
import pendulum
from urllib.parse import urlparse
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from airflow.models.dag import DAG
from airflow.models.param import Param
from airflow.providers.databricks.hooks.databricks import DatabricksHook
from airflow.providers.microsoft.azure.hooks.wasb import WasbHook

from databricks_operators import DatabricksRunNotebookOperator


with DAG(
    dag_id="train_and_evaluate",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None,
    tags=["mlops", "llm"],
    params={
        "source_blob_url": Param(
            "azure://raw-data/python_dedupe_definitions_v2.pkl",
            type="string",
            title="Source Azure Blob URL",
            description="URL to the source data file in format: azure://<container>/<blob_name>",
        ),
        "num_pretrain": Param(1000, type="integer", title="Pre-training Samples", description="Number of samples for pre-training dataset. Use -1 for all."),
        "num_instruct": Param(1000, type="integer", title="Instruction-tuning Samples", description="Number of samples for instruction-tuning dataset. Use -1 for all."),
        "base_model_name": Param(
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            type="string",
            title="Base Model Name",
            description="The name of the pre-trained model to use as a base for fine-tuning.",
            enum=[
                "deepseek-ai/deepseek-coder-1.3b-instruct"
            ],
        ),
        "epochs": Param(1, type="integer", title="Epochs", description="Number of training epochs for both pre-training and fine-tuning."),
        "learning_rate": Param(5e-5, type="number", title="Learning Rate", description="Learning rate for both pre-training and fine-tuning."),
    },
) as dag:
    NEW_MODEL_NAME = "mlops_coder"

    @task
    def download_raw_data(blob_url: str) -> str:
        """
        Downloads a file from Azure Blob Storage, overwriting any existing local version.
        The file is saved in a temporary directory under its original name.
        """
        print(f"Parsing blob URL: '{blob_url}'...")
        parsed_url = urlparse(blob_url)
        if parsed_url.scheme != "azure":
            raise ValueError(f"Invalid blob URL scheme: '{parsed_url.scheme}'. Must be 'azure'.")

        container_name = parsed_url.netloc
        blob_name = parsed_url.path.lstrip("/")
        base_blob_name = os.path.basename(blob_name)

        if not container_name or not blob_name:
            raise ValueError(f"Invalid blob URL format: '{blob_url}'. Could not extract container or blob name.")

        hook = WasbHook(wasb_conn_id="azure_storage")
        blob_service_client = hook.get_conn()
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Construct local path in a temporary directory with the original filename
        local_path = os.path.join(tempfile.gettempdir(), base_blob_name)
        print(f"Local file path is set to: {local_path}")
        
        print(f"Downloading '{blob_name}' from container '{container_name}'...")
        with open(local_path, "wb") as file_handle:
            downloader = blob_client.download_blob(max_concurrency=4)
            downloader.readinto(file_handle)

        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Successfully downloaded file of size {file_size_mb:.2f} MB.")
        print(f"File available for inspection at: {local_path}")
        return local_path

    @task
    def extract_dataset(pickle_path: str, num_pretrain: int, num_instruct: int) -> dict:
        """
        Loads a pickle file and converts its content into two separate JSONL files:
        one for pre-training and one for instruction-tuning, based on the logic
        from convert.py.
        """
        num_pretrain = int(num_pretrain)
        num_instruct = int(num_instruct)

        print(f"Loading data from pickle file: {pickle_path}")
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded {len(data):,} records.")

        DOCSTRING_KEY = 'docstring'
        LANGUAGE_KEY = 'language'
        CODE_KEY = 'function'
        TARGET_LANGUAGE = 'python'

        pretrain_filter = lambda r: r.get(LANGUAGE_KEY) == TARGET_LANGUAGE and not r.get(DOCSTRING_KEY) and r.get(CODE_KEY)
        pretrain_transform = lambda r: {"code": r[CODE_KEY]}

        instruct_filter = lambda r: r.get(LANGUAGE_KEY) == TARGET_LANGUAGE and r.get(DOCSTRING_KEY) and r.get(CODE_KEY)
        instruct_transform = lambda r: {"docstring": r[DOCSTRING_KEY], "code": r[CODE_KEY]}

        def _process_and_write(output_filename, filter_logic, transform_logic, num_samples):
            output_dir = os.path.dirname(pickle_path)
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"Processing data for '{output_filename}'...")
            candidates = [transform_logic(r) for r in data if filter_logic(r)]
            
            if not candidates:
                print(f"No candidates found for '{output_filename}'. Skipping file creation.")
                return None
                
            if num_samples is not None and num_samples >= 0:
                samples = candidates[:num_samples]
            else: # Use all if num_samples is negative
                samples = candidates

            print(f"Writing {len(samples):,} samples to '{output_path}'")
            with open(output_path, "w", encoding="utf-8") as f:
                for item in samples:
                    f.write(json.dumps(item) + "\n")
            return output_path

        pretrain_path = _process_and_write(
            "dataset_pretrain.jsonl", pretrain_filter, pretrain_transform, num_pretrain
        )
        instruct_path = _process_and_write(
            "dataset_instruction.jsonl", instruct_filter, instruct_transform, num_instruct
        )
        
        print("Conversion process finished.")
        return {"pretrain_file": pretrain_path, "instruction_file": instruct_path}

    download_task = download_raw_data(
        blob_url="{{ params.source_blob_url }}",
    )

    conversion_task = extract_dataset(
        pickle_path=download_task,
        num_pretrain="{{ params.num_pretrain }}",
        num_instruct="{{ params.num_instruct }}",
    )

    @task
    def upload_dataset(processed_files: dict):
        """
        Uploads the processed JSONL dataset files to Databricks DBFS
        using the dbfs/create, dbfs/add-block, and dbfs/close APIs.
        """
        print("Uploading processed dataset files to Databricks DBFS...")
        hook = DatabricksHook(databricks_conn_id="azure_databricks")
        
        uploaded_paths = {}
        for file_type, local_path in processed_files.items():
            if not local_path or not os.path.exists(local_path):
                print(f"Skipping '{file_type}': file path is missing or file does not exist.")
                continue

            file_name = os.path.basename(local_path)
            dbfs_path = f"/tmp/{file_name}"
            
            print(f"Uploading '{local_path}' to DBFS path '{dbfs_path}'...")
            
            # 1. Create a handle for the file upload
            create_handle_response = hook._do_api_call(
                ("POST", "2.0/dbfs/create"),
                json={"path": dbfs_path, "overwrite": "true"},
            )
            handle = create_handle_response["handle"]
            print(f"Successfully created a handle for '{file_name}': {handle}")

            # 2. Add content in blocks
            with open(local_path, "rb") as f:
                while True:
                    chunk = f.read(1 * 1024 * 1024)  # Read in 1MB chunks
                    if not chunk:
                        break
                    encoded_chunk = base64.b64encode(chunk).decode("utf-8")
                    hook._do_api_call(
                        ("POST", "2.0/dbfs/add-block"),
                        json={"handle": handle, "data": encoded_chunk},
                    )
            print(f"Finished writing all blocks for '{file_name}'.")

            # 3. Close the handle
            hook._do_api_call(
                ("POST", "2.0/dbfs/close"),
                json={"handle": handle},
            )

            print(f"Successfully uploaded '{file_name}' to DBFS.")
            # The returned path must be in the format dbfs:/... for Spark to use it.
            uploaded_paths[file_type] = f"dbfs:{dbfs_path}"
        
        return uploaded_paths

    upload_task = upload_dataset(conversion_task)

    pretrain_model_task = DatabricksRunNotebookOperator(
        task_id="pretrain_model",
        databricks_conn_id="azure_databricks",
        json={
            "run_name": f"pre_train_{NEW_MODEL_NAME}",
            "tasks": [
                {
                    "task_key": "pretrain_model",
                    "notebook_task": {
                        "notebook_path": "/Workspace/Volumes/mlops/default/notebooks/mlflow/pretrain",
                        "base_parameters": {
                            "BASE_MODEL_NAME": "{{ params.base_model_name }}",
                            "NEW_MODEL_NAME": NEW_MODEL_NAME,
                            "DATASET_PATH": "{{ ti.xcom_pull(task_ids='upload_dataset')['pretrain_file'] }}",
                            "EPOCHS": "{{ params.epochs }}",
                            "LEARNING_RATE": "{{ params.learning_rate }}",
                        },
                    },
                }
            ],
        },
    )

    train_model_task = DatabricksRunNotebookOperator(
        task_id="train_model",
        databricks_conn_id="azure_databricks",
        json={
            "run_name": "fine_tune_{{ ti.xcom_pull(task_ids='pretrain_model')['model'] }}",
            "tasks": [
                {
                    "task_key": "train_model",
                    "notebook_task": {
                        "notebook_path": "/Workspace/Volumes/mlops/default/notebooks/mlflow/train",
                        "base_parameters": {
                            "BASE_MODEL_NAME": "models:/{{ ti.xcom_pull(task_ids='pretrain_model')['model'] }}/{{ ti.xcom_pull(task_ids='pretrain_model')['version'] }}",
                            "NEW_MODEL_NAME": NEW_MODEL_NAME,
                            "DATASET_PATH": "{{ ti.xcom_pull(task_ids='upload_dataset')['instruction_file'] }}",
                            "EPOCHS": "{{ params.epochs }}",
                            "LEARNING_RATE": "{{ params.learning_rate }}",
                        },
                    },
                }
            ],
        },
    )

    evaluate_model_task = DatabricksRunNotebookOperator(
        task_id="evaluate_model",
        databricks_conn_id="azure_databricks",
        json={
            "run_name": "evaluate_{{ ti.xcom_pull(task_ids='train_model')['model'] }}_v{{ ti.xcom_pull(task_ids='train_model')['version'] }}",
            "tasks": [
                {
                    "task_key": "evaluate_model",
                    "notebook_task": {
                        "notebook_path": "/Workspace/Volumes/mlops/default/notebooks/mlflow/evaluate",
                        "base_parameters": {
                            "BASE_MODEL_NAME": "{{ params.base_model_name }}",
                            "NEW_MODEL_NAME": NEW_MODEL_NAME,
                            "CHALLENGER_VERSION": "{{ ti.xcom_pull(task_ids='train_model')['version'] }}",
                        },
                    },
                }
            ],
        },
    )

    @task
    def stage_model(evaluation_output: dict, challenger_version: str):
        """
        Compares challenger model score with the champion. If better, transitions
        the challenger model to the 'Staging' stage in the MLflow Model Registry
        using the Databricks API. If the score is not better, the task is skipped.
        """
        print(f"Received evaluation results: {evaluation_output}")
        challenger_score = evaluation_output["challenger_score"]
        champion_score = evaluation_output.get("champion_score", 0)

        if challenger_score > champion_score:
            print(
                f"Challenger model is better (Score: {challenger_score} > "
                f"Champion Score: {champion_score})."
            )
            print(
                f"Transitioning model '{NEW_MODEL_NAME}' version {challenger_version} "
                f"to 'Staging' stage via Databricks API."
            )

            hook = DatabricksHook(databricks_conn_id="azure_databricks")
            hook._do_api_call(
                ("POST", "2.0/mlflow/model-versions/transition-stage"),
                json={
                    "name": NEW_MODEL_NAME,
                    "version": challenger_version,
                    "stage": "Staging",
                    "archive_existing_versions": False,
                },
            )
            print("Successfully transitioned model to 'Staging' stage via Databricks API.")
        else:
            message = (
                f"Challenger model is not better (Score: {challenger_score} <= "
                f"Champion Score: {champion_score}). Skipping stage transition."
            )
            print(message)
            raise AirflowSkipException(message)

    stage_task = stage_model(
        evaluation_output=evaluate_model_task.output,
        challenger_version="{{ ti.xcom_pull(task_ids='train_model')['version'] }}",
    )
    
    download_task >> conversion_task >> upload_task >> pretrain_model_task >> train_model_task
    train_model_task >> evaluate_model_task >> stage_task
