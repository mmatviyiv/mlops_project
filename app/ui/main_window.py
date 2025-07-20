from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PyQt5.QtCore import Qt

from app.utils.databricks_api import DatabricksAPI
from app.utils.config import Config
from app.ui.status_tab import StatusTab
from app.ui.settings_tab import SettingsTab


class MainWindow(QWidget):
    def __init__(self, download_threadpool, api_server_manager, parent=None):
        super().__init__(parent)
        self.title = 'Refactoring Assistant'
        self.left = 100
        self.top = 100
        self.width = 480
        self.height = 480
 
        self.config = Config()
        self.api_client = DatabricksAPI(
            token=self.config.token,
            host=self.config.host,
            parent=self
        )

        self.download_threadpool = download_threadpool
        self.api_server_manager = api_server_manager
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setFixedSize(self.width, self.height)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        self.status_tab = StatusTab(self.api_server_manager, self.config)
        self.settings_tab = SettingsTab(
            self.api_client, self.config, self.download_threadpool
        )
        self.settings_tab.load_settings()

        tabs.addTab(self.status_tab, 'Status')
        tabs.addTab(self.settings_tab, 'Settings')
        
        tabs.currentChanged.connect(self.on_tab_changed)
        
        layout.addWidget(tabs)
        self.setLayout(layout)

    def on_tab_changed(self, index):
        self.status_tab.refresh_model_list()