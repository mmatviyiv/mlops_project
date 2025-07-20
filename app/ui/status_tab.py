from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QTextEdit, QHBoxLayout, QComboBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
import time


class StatusTab(QWidget):
    def __init__(self, api_server_manager, config, parent=None):
        super().__init__(parent)
        self.api_server_manager = api_server_manager
        self.config = config
        self.start_time = None
        self.uptime_timer = QTimer(self)
        self.initUI()
        self.connect_signals()
        self.refresh_model_list()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        # --- Top 2x2 Grid ---
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        
        self.api_status_label = QLabel("API: checking...")
        self.api_status_label.setStyleSheet("color: orange; font-weight: bold;")
        
        self.server_address_label = QLabel()
        self.server_address_label.setVisible(False)
        self.server_address_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.model_dropdown = QComboBox()
        self.restart_button = QPushButton("Restart")
        self.restart_button.setEnabled(False)

        # Add widgets to the grid
        grid_layout.addWidget(self.api_status_label, 0, 0)
        grid_layout.addWidget(self.model_dropdown, 0, 1)
        grid_layout.addWidget(self.server_address_label, 1, 0, alignment=Qt.AlignTop)
        grid_layout.addWidget(self.restart_button, 1, 1, alignment=Qt.AlignRight | Qt.AlignTop)

        # Set column stretch to make dropdown and button align nicely
        grid_layout.setColumnStretch(0, 1)
        
        main_layout.addLayout(grid_layout)

        # --- Logs ---
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.log_box)
        main_layout.setStretchFactor(self.log_box, 1)

    def connect_signals(self):
        self.restart_button.clicked.connect(self.handle_restart)
        self.api_server_manager.log_received.connect(self.update_logs)
        self.api_server_manager.server_started.connect(self.on_server_started)
        self.api_server_manager.server_stopped.connect(self.on_server_stopped)
        self.model_dropdown.currentIndexChanged.connect(self.on_model_selection_change)
        self.uptime_timer.timeout.connect(self.update_uptime)

    def handle_restart(self):
        self.log_box.clear()
        self.api_server_manager.restart_server()

    def on_model_selection_change(self, index):
        if index >= 0:
            model_name = self.model_dropdown.currentText()
            self.config.selected_model = model_name

    def update_logs(self, message):
        self.log_box.append(message)

    def _format_uptime(self, total_seconds):
        if total_seconds < 60:
            return "<1m"
        
        hours, remainder = divmod(total_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        if hours >= 1:
            return f"~{int(hours)}h {int(minutes)}m"
        return f"~{int(minutes)}m"

    def on_server_started(self, host, port):
        self.api_status_label.setStyleSheet("color: green; font-weight: bold;")
        self.start_uptime_timer()
        self.update_uptime()
        self.restart_button.setEnabled(True)
        
        address = f"http://{host}:{port}"
        self.server_address_label.setText(address)
        self.server_address_label.setVisible(True)

    def on_server_stopped(self):
        self.api_status_label.setText("API: down")
        self.api_status_label.setStyleSheet("color: red; font-weight: bold;")
        self.stop_uptime_timer()
        self.restart_button.setEnabled(True)
        self.server_address_label.setVisible(False)

    def start_uptime_timer(self):
        self.start_time = time.time()
        self.uptime_timer.start(1000)

    def stop_uptime_timer(self):
        self.uptime_timer.stop()

    def update_uptime(self):
        if self.start_time:
            elapsed_seconds = int(time.time() - self.start_time)
            self.api_status_label.setText(f"API: up for {self._format_uptime(elapsed_seconds)}")
        else:
            self.api_status_label.setText("API: down")

    def refresh_model_list(self):
        current_model = self.model_dropdown.currentText()
        self.model_dropdown.blockSignals(True)
        self.model_dropdown.clear()
        
        downloaded_models = self.config.downloaded_models
        if downloaded_models:
            self.model_dropdown.addItems(downloaded_models)
        
        initial_model = self.config.model or (downloaded_models[0] if downloaded_models else None)

        index = self.model_dropdown.findText(initial_model)
        self.model_dropdown.blockSignals(False)
        if initial_model and index >= 0:
            self.model_dropdown.setCurrentIndex(index)
        
        self.on_model_selection_change(self.model_dropdown.currentIndex())
