import os
import re
import requests
import mlflow
from PyQt5.QtCore import QObject, pyqtSignal
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository


class DatabricksAPI(QObject):
    """
    Handles all communication with the Databricks and MLflow REST APIs.
    """
    validation_finished = pyqtSignal(bool, str)
    models_fetched = pyqtSignal(list, str)

    def __init__(self, host, token, parent=None):
        super().__init__(parent)

        self.host = host.rstrip('/')
        self.token = token
        self.headers = {'Authorization': f'Bearer {self.token}'}
        
        os.environ['DATABRICKS_HOST'] = self.host
        os.environ['DATABRICKS_TOKEN'] = self.token
        self.client = mlflow.tracking.MlflowClient(
            tracking_uri="databricks",
            registry_uri="databricks"
        )

    def validate_credentials(self, host, token):
        api_url = f"{host}/api/2.0/mlflow/registered-models/list"
        try:
            headers = {'Authorization': f'Bearer {token}'}
            response = requests.get(api_url, headers=headers, timeout=5)
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.exceptions.RequestException as e:
            return False


    def list_models(self, channel):
        api_url = f"{self.host}/api/2.0/mlflow/registered-models/list"

        response = requests.get(api_url, headers=self.headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        models_list = []

        if "registered_models" not in data:
            return []
        
        for model in data["registered_models"]:
            if "latest_versions" not in model:
                continue
            
            for version in model["latest_versions"]:
                if version.get("current_stage") != channel:
                    continue

                models_list.append(version)

        return models_list

    def list_artifacts(self, run_id, path=""):
        artifact_url = f"{self.host}/api/2.0/mlflow/artifacts/list?run_uuid={run_id}&path={path}"
        response = requests.get(artifact_url, headers=self.headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        artifacts = []
        for artifact in data["files"]:
            if artifact["is_dir"]:
                artifacts.extend(self.list_artifacts(run_id, artifact["path"]))
            else:
                artifacts.append(artifact)

        return artifacts

    def download_artifact(self, source, artifact_path):
        artifact_path = re.sub(r'^model/', '', artifact_path)
        chunk_size = 100 * 1024**2

        repository = DatabricksArtifactRepository(source)
        read_credentials = repository._get_read_credential_infos(artifact_path)
        cloud_credential_info = read_credentials[0]
        cloud_headers = repository._extract_headers_from_credentials(cloud_credential_info.headers)

        with requests.get(
            cloud_credential_info.signed_uri,
            headers=cloud_headers, stream=True
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=chunk_size):
                yield chunk
