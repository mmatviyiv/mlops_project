import signal
import sys
from pathlib import Path
from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QApplication

# Dirty hack to allow absolute imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.ui.main_window import MainWindow
from app.utils.api_server_manager import ApiServerManager
from app.utils.config import Config

def main():
    app = QApplication(sys.argv)
    
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    download_threadpool = QThreadPool()
    download_threadpool.setMaxThreadCount(2)
    
    config = Config()
    api_server_manager = ApiServerManager(config)
    
    main_window = MainWindow(download_threadpool, api_server_manager)
    main_window.show()

    api_server_manager.start_server(config.model)

    app.aboutToQuit.connect(api_server_manager.stop_server)
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
