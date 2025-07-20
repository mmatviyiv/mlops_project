import os
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot


class ModelDownloaderSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    progress = pyqtSignal(int)


class ModelDownloader(QRunnable):
    def __init__(self, api_client, model_data):
        super(ModelDownloader, self).__init__()
        self.api_client = api_client
        self.run_id = model_data['run_id']
        self.source = model_data['source']
        self.model_name = model_data['name']
        self.model_version = model_data['version']

        model_folder = f"{self.model_name}_{self.model_version}"
        self.model_path_storage = Path(__file__).parent.parent / 'models' / model_folder
        self.model_path_loading = f"{self.model_path_storage.resolve()}_loading"
        self.signals = ModelDownloaderSignals()

    @pyqtSlot()
    def run(self):
        os.makedirs(self.model_path_loading, exist_ok=True)

        artifacts = self.api_client.list_artifacts(run_id=self.run_id)
        size_to_download = sum(a['file_size'] for a in artifacts)
        size_downloaded = 0

        for artifact in artifacts:
            file_path = f"{self.model_path_loading}/{artifact['path']}"
            file_dir = "/".join(file_path.split("/")[:-1])
            os.makedirs(file_dir, exist_ok=True)

            with open(file_path, 'wb') as f:
                for chunk in self.api_client.download_artifact(self.source, artifact['path']):
                    f.write(chunk)
                    size_downloaded += len(chunk)

                    progress = int((size_downloaded / size_to_download) * 100)
                    self.signals.progress.emit(progress)

        os.rename(self.model_path_loading, self.model_path_storage)
        self.signals.finished.emit()
