from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QGridLayout, QLabel, QLineEdit, 
                             QPushButton, QHBoxLayout, QComboBox, 
                             QApplication, QMessageBox, QGroupBox,
                             QListWidget, QListWidgetItem, QVBoxLayout)

from app.utils.model_downloader import ModelDownloader


class SettingsTab(QWidget):
    def __init__(self, api_client, config, download_threadpool, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.config = config
        self.download_threadpool = download_threadpool
        
        self.initUI()
        self.connect_signals()
    
    def initUI(self):
        grid_layout = QGridLayout(self)
        grid_layout.setSpacing(10)

        grid_layout.addWidget(QLabel("Registry:"), 0, 0, alignment=Qt.AlignTop)

        # Host and Token
        host_label = QLabel("Host:")
        self.host_input = QLineEdit()
        token_label = QLabel("Token:")
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        self.save_button = QPushButton("Save")
        self.save_button.setFixedWidth(100)
        self.save_button.setEnabled(False)

        registry_layout = QGridLayout()
        registry_layout.addWidget(host_label, 0, 0)
        registry_layout.addWidget(self.host_input, 0, 1)
        registry_layout.addWidget(token_label, 1, 0)
        registry_layout.addWidget(self.token_input, 1, 1)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.save_button)
        registry_layout.addLayout(button_layout, 2, 1)
        grid_layout.addLayout(registry_layout, 1, 0, 1, 2)

        # --- Model Management Group ---
        self.model_management_group = QGroupBox()
        model_layout = QGridLayout(self.model_management_group)
        model_layout.setColumnStretch(1, 1)
        
        model_layout.addWidget(QLabel("Channel:"), 0, 0)
        self.channel_dropdown = QComboBox()
        self.channel_dropdown.addItems(["Production", "Staging"])
        model_layout.addWidget(self.channel_dropdown, 0, 1)
        
        self.model_list = QListWidget()
        self.download_button = QPushButton("Download")
        self.download_button.setEnabled(False)
        self.download_button.setDefault(True)
        self.refresh_button = QPushButton("Refresh")

        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(self.model_list)
        
        button_vstack = QVBoxLayout()
        button_vstack.addWidget(self.refresh_button)
        button_vstack.addWidget(self.download_button)
        button_vstack.addStretch(1)
        model_selection_layout.addLayout(button_vstack)

        model_layout.addLayout(model_selection_layout, 1, 0, 1, 2)
        
        self.loader_status_label = QLabel("Idle")
        self.loader_status_label.setVisible(False)
        model_layout.addWidget(self.loader_status_label, 2, 0, 1, 2)

        grid_layout.addWidget(self.model_management_group, 2, 0, 1, 2)
        grid_layout.setRowStretch(3, 1)

    def connect_signals(self):
        self.host_input.textChanged.connect(self.on_settings_changed)
        self.token_input.textChanged.connect(self.on_settings_changed)
        self.save_button.clicked.connect(self.save_settings)
        self.channel_dropdown.currentIndexChanged.connect(self.save_channel_selection)
        self.refresh_button.clicked.connect(self.refresh_models)
        self.model_list.currentItemChanged.connect(self.on_model_selection_changed)
        self.download_button.clicked.connect(self.handle_model_select)

    def handle_model_select(self):
        selected_item = self.model_list.currentItem()
        if not selected_item:
            return

        model_data = selected_item.data(Qt.UserRole)
        
        downloader = ModelDownloader(self.api_client, model_data)
        downloader.signals.progress.connect(self.update_loader_progress)
        downloader.signals.finished.connect(self.on_loading_finished)
        downloader.signals.error.connect(self.on_loading_error)

        self.download_threadpool.start(downloader)
        self.set_ui_loading_state(True)

    def set_ui_loading_state(self, is_loading):
        self.download_button.setEnabled(not is_loading)
        self.refresh_button.setEnabled(not is_loading)
        self.model_list.setEnabled(not is_loading)
        self.channel_dropdown.setEnabled(not is_loading)
        
        self.loader_status_label.setVisible(is_loading)
        if is_loading:
            self.loader_status_label.setText("Downloading...")
        
    def update_loader_progress(self, progress):
        self.loader_status_label.setText(f"Downloading... {progress}%")

    def on_loading_finished(self):
        self.set_ui_loading_state(False)
        QMessageBox.information(self, "Success", "Model downloaded successfully.")
        self.refresh_models()

    def on_loading_error(self, error_tuple):
        self.set_ui_loading_state(False)
        error_message = f"Error downloading model: {error_tuple[1]}"
        QMessageBox.critical(self, "Download Error", error_message)

    def on_model_selection_changed(self):
        is_item_selected = self.model_list.currentItem() is not None
        self.download_button.setEnabled(is_item_selected)

    def on_settings_changed(self):
        host_changed = self.host_input.text() != self.config.host
        token_changed = self.token_input.text() != self.config.token
        self.save_button.setEnabled(host_changed or token_changed)

    def set_model_controls_visibility(self, visible):
        self.model_management_group.setVisible(visible)

    def save_settings(self):
        host = self.host_input.text()
        token = self.token_input.text()
        
        self.save_button.setEnabled(False)
        QApplication.processEvents()
        
        self.api_client.update_credentials(host, token)
        success = self.api_client.validate_credentials()
        
        if success:
            self.config.host = host
            self.config.token = token
            QMessageBox.information(self, "Success", "Credentials are valid.")
        else:
            QMessageBox.warning(self, "Validation Failed", "Invalid credentials")
        
        self.save_button.setEnabled(True)
    
    def save_channel_selection(self):
        self.config.channel = self.channel_dropdown.currentText()
        self.refresh_models()

    def refresh_models(self):
        self.model_list.clear()
        self.download_button.setEnabled(False)

        channel = self.config.channel
        if not channel:
            return
            
        models_from_api = self.api_client.list_models(channel)
        downloaded_models = self.config.downloaded_models
        
        for model in models_from_api:
            # Construct the model folder name as it appears in the local directory
            model_folder_name = f"{model['name']}_{model['version']}"
            if model_folder_name in downloaded_models:
                continue

            display_text = f"{model['name']}/{model['version']}"
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, model)
            self.model_list.addItem(item)

    def load_settings(self):
        self.host_input.setText(self.config.host)
        self.token_input.setText(self.config.token)

        saved_channel = self.config.channel
        if saved_channel:
            index = self.channel_dropdown.findText(saved_channel)
            if index >= 0:
                self.channel_dropdown.setCurrentIndex(index)
        
        self.set_model_controls_visibility(bool(self.config.host and self.config.token))
        if self.config.host and self.config.token:
            self.refresh_models()
