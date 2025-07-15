from __future__ import annotations

import pendulum
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.models.param import Param
from airflow.providers.databricks.hooks.databricks import DatabricksHook
from urllib.parse import quote


def get_staging_models() -> list[str]:
    """
    Fetches model versions from the MLflow Model Registry that are in the
    'Staging' stage and formats them for the Airflow UI. If no models are
    found or an error occurs, an empty list is returned to disable the selector.
    
    Returns a list of strings like: 'model_name/123'
    """
    try:
        hook = DatabricksHook(databricks_conn_id="azure_databricks")
        
        registered_models_resp = hook._do_api_call(("GET", "api/2.0/mlflow/registered-models/search"))
        registered_models = registered_models_resp.get("registered_models", [])
        
        if not registered_models:
            print("No registered models found.")
            return []

        staging_models = []
        for model in registered_models:
            model_name = model['name']
            
            filter_str = f"name='{model_name}'"
            encoded_filter = quote(filter_str)
            endpoint = f"api/2.0/mlflow/model-versions/search?filter={encoded_filter}"
            
            versions_resp = hook._do_api_call(("GET", endpoint))
            versions = versions_resp.get("model_versions", [])

            for version in versions:
                if version.get("current_stage") == "Staging":
                    display_name = f"{version['name']}/{version['version']}"
                    staging_models.append(display_name)
        
        if not staging_models:
            print("No models in 'Staging' stage found.")
            return []

        return staging_models
    except Exception as e:
        print(f"Failed to fetch models from Databricks: {e}")
        return []


with DAG(
    dag_id="release",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None,
    tags=["mlops", "llm", "promotion"],
    params={
        "model_to_promote": Param(
            "NO_MODELS_AVAILABLE",
            type="string",
            title="Staging Model Version",
            description="Select the model version to promote. If this list is empty, no models are in the 'Staging' stage.",
            enum=get_staging_models(),
        ),
    },
) as dag:

    @task
    def promote_model(model_to_promote: str):
        """
        Promotes the selected model version to 'Production' and archives the old one.
        """
        print(f"Received selection: {model_to_promote}")

        if model_to_promote == "NO_MODELS_AVAILABLE":
            raise ValueError("No model was selected for promotion. Please ensure a model is in 'Staging'.")

        if "/" not in model_to_promote:
            raise ValueError(f"Invalid model format: '{model_to_promote}'. Expected 'model_name/version'.")
        
        model_name, model_version = model_to_promote.rsplit('/', 1)
        print(f"Parsed model name: '{model_name}', version: '{model_version}'")
        
        hook = DatabricksHook(databricks_conn_id="azure_databricks")

        print(f"Transitioning model '{model_name}' version {model_version} to 'Production'...")
        hook._do_api_call(
            ("POST", "api/2.0/mlflow/model-versions/transition-stage"),
            json={
                "name": model_name,
                "version": model_version,
                "stage": "Production",
                "archive_existing_versions": True,
            }
        )
        print("Successfully promoted new model to 'Production'.")

    promote_model(model_to_promote="{{ params.model_to_promote }}") 