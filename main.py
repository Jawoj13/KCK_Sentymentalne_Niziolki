import sys
import queue
from PyQt5.QtWidgets import QApplication
from backend import CameraWorker, SyncInferenceWorker
from ui import MainWindow

# Klasa koordynująca pracę interfejsu oraz wątków tła.
class AppController:
    def __init__(self):
        self.window = MainWindow()

        # Inicjalizacja kolejek o małym rozmiarze.
        # maxsize=2 gwarantuje, że nie przetwarzamy starych klatek, jeśli YOLO zwolni.
        self.queue_a = queue.Queue(maxsize=2)
        self.queue_b = queue.Queue(maxsize=2)

        # Uruchomienie wątku zajmującego się rozpoznawaniem.
        self.inference_worker = SyncInferenceWorker(self.queue_a, self.queue_b)
        # Połączenie sygnału z wątku YOLO do funkcji odświeżającej okno UI.
        self.inference_worker.frames_ready.connect(self.window.update_both_labels)
        self.inference_worker.start()

        # Kamera laptopa
        self.worker_a = CameraWorker(0, self.queue_a)
        self.worker_a.start()

        # Kamera z telefonu
        self.worker_b = None

        self.window.connect_btn.clicked.connect(self.connect_ip_camera)

    # Metoda obsługująca dynamiczne łączenie się z nowym strumieniem wideo.
    def connect_ip_camera(self):
        if self.worker_b is not None:
            self.worker_b.stop()

        stream_url = self.window.ip_input.text()

        self.worker_b = CameraWorker(stream_url, self.queue_b)
        self.worker_b.start()

    def cleanup(self):
        self.worker_a.stop()
        if self.worker_b is not None:
            self.worker_b.stop()
        self.inference_worker.stop()


def main():
    app = QApplication(sys.argv)
    controller = AppController()
    controller.window.show()

    app.aboutToQuit.connect(controller.cleanup)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()