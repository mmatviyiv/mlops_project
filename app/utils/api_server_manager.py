from PyQt5.QtCore import QObject, pyqtSignal, QProcess, QTimer, QProcessEnvironment
import logging
import requests
from pathlib import Path


class ApiServerManager(QObject):
    log_received = pyqtSignal(str)
    server_started = pyqtSignal(str, int)
    server_stopped = pyqtSignal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.process = None
        
        self.health_check_timer = QTimer(self)
        self.health_check_timer.timeout.connect(self.check_server_health)
        self.health_check_attempts = 0
        
        self.kill_timer = QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.timeout.connect(self._force_kill_server)

        self._is_restarting = False
        self._restart_model_name = None

    def start_server(self, model_name):
        if self.process and self.process.state() == QProcess.Running:
            logging.warning("Server is already running.")
            return

        self.health_check_attempts = 0

        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self._on_process_finished)

        models_dir = Path(__file__).parent.parent / 'models'
        model_name_fs = '_'.join(model_name.split('/'))
        model_path = models_dir / model_name_fs / "model" / "model"
        
        env = QProcessEnvironment.systemEnvironment()
        env.insert("MODEL_NAME", model_name)
        env.insert("MODEL_PATH", str(model_path))
        
        self.process.setProcessEnvironment(env)

        command = "uvicorn"
        api_path = "app.api.main:app"
        app_dir = str(Path(__file__).parent.parent.parent)
        
        args = [
            api_path,
            "--host", self.config.api_host,
            "--port", str(self.config.api_port),
            "--app-dir", app_dir
        ]

        self.log_received.emit(f"Starting server: {command} {' '.join(args)}")
        self.process.start(command, args)
        self.health_check_timer.start(2000)

    def stop_server(self):
        self.health_check_timer.stop()
        if self.process and self.process.state() == QProcess.Running:
            self.log_received.emit("Stopping server...")
            self.process.terminate()
            self.kill_timer.start(3000)
        else:
            self._on_process_finished()

    def _force_kill_server(self):
        if self.process and self.process.state() == QProcess.Running:
            self.log_received.emit("Server did not terminate gracefully. Killing...")
            self.process.kill()

    def restart_server(self):
        model_name = self.config.model
        if not model_name:
            self.log_received.emit("Cannot restart: No model selected in config.")
            return
            
        self.log_received.emit("Restarting server...")
        self._is_restarting = True
        self._restart_model_name = model_name
        self.stop_server()
            
    def _on_process_finished(self):
        self.kill_timer.stop()
        self.server_stopped.emit()

        if self._is_restarting:
            model_to_start = self._restart_model_name
            self._is_restarting = False
            self._restart_model_name = None
            if model_to_start:
                self.start_server(model_to_start)

    def check_server_health(self):
        if not self.process or self.process.state() != QProcess.Running:
            self.health_check_timer.stop()
            return
        
        self.health_check_attempts += 1
        self.log_received.emit(f"Health check attempt #{self.health_check_attempts}...")

        try:
            host = self.config.api_host
            port = self.config.api_port
            response = requests.get(f"http://{host}:{port}/health", timeout=1)

            if response.status_code == 200:
                self.log_received.emit("Health check successful. Server is up.")
                self.health_check_timer.stop()
                self.server_started.emit(host, port)
        except requests.ConnectionError:
            if self.health_check_attempts >= 10:
                self.log_received.emit("Health check failed after multiple attempts. Stopping check.")
                self.health_check_timer.stop()
                self.server_stopped.emit()
    
    def handle_stdout(self):
        if not self.process:
            return
        message = self.process.readAllStandardOutput().data().decode().strip()
        if message:
            self.log_received.emit(message) 