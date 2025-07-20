import json
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal


class Config(QObject):
    config_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__()
        self._path = Path(__file__).parent.parent / 'config.json'
        self._fields = ['host', 'token', 'channel', 'model']
        self._data = self._load()

        # API Server configuration
        self.api_host = "0.0.0.0"
        self.api_port = 2525

    def _load(self) -> dict:
        try:
            return json.loads(self._path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self):
        try:
            self._path.write_text(json.dumps(self._data, indent=4))
        except IOError as e:
            print(f"Error saving config: {e}")

    def __getattr__(self, name):
        if name in self._fields:
            return self._data.get(name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if '_fields' in self.__dict__ and name in self._fields:
            self._data[name] = value
            self._save()
            self.config_changed.emit()
        else:
            super().__setattr__(name, value)
    
    @property
    def all(self) -> dict:
        return self._data.copy()

    @property
    def downloaded_models(self) -> list:
        models_dir = Path(__file__).parent.parent / 'models'
        if not models_dir.is_dir():
            return []
        
        models = []
        for model_path in models_dir.iterdir():
            if model_path.is_dir() and not model_path.name.endswith('_loading'):
                models.append(model_path.name)
        
        return models
