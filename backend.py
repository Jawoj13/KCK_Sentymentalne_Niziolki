import time
import queue
import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from ultralytics import YOLO

# Przechwytuje strumienie wideo z kamery.
# Wykorzystuje QThread, aby proces dekodowania klatek nie blokował interfejsu użytkownika
class CameraWorker(QThread):
    def __init__(self, stream_url, frame_queue):
        super().__init__()
        self.stream_url = stream_url  # Adres URL (IP) lub indeks kamery (0)
        self.frame_queue = frame_queue
        self._is_running = True

    def _connect(self):
        import sys
        if isinstance(self.stream_url, int):
            if sys.platform.startswith('win'):
                backend = cv2.CAP_DSHOW
            elif sys.platform.startswith('linux'):
                backend = cv2.CAP_V4L2
            else:
                backend = cv2.CAP_ANY

            indices_to_test = [self.stream_url] + [i for i in range(10) if i != self.stream_url]

            for index in indices_to_test:
                test_capture = cv2.VideoCapture(index, backend)
                if test_capture.isOpened():
                    ret, _ = test_capture.read()
                    if ret:
                        return test_capture
                test_capture.release()
            return None
        else:
            capture = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if capture.isOpened():
                return capture
            return None

    # Główna pętla wątku odpowiedzialna za odczyt obrazu z OpenCV
    def run(self):
        while self._is_running:
            capture = self._connect()

            if capture is None:
                self.msleep(1000)
                continue

            while self._is_running and capture.isOpened():
                ret, frame = capture.read()
                if ret:
                    timestamp = time.time()
                    try:
                        # Próba umieszczenia klatki w kolejce.
                        self.frame_queue.put_nowait((timestamp, frame))
                    except queue.Full:
                        pass
                else:
                    break

            if capture:
                capture.release()

    def stop(self):
        self._is_running = False
        self.wait()

# Klasa przetwarzająca,msynchronizuje obrazy z dwóch źródeł i wykonuje detekcję YOLO.
class SyncInferenceWorker(QThread):
    frames_ready = pyqtSignal(QImage, QImage)

    def __init__(self, queue_a, queue_b):
        super().__init__()
        self.queue_a = queue_a
        self.queue_b = queue_b
        self._is_running = True
        self.model = None
        self.sync_threshold = 0.05

    def _np_to_qimage(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = channel * width
        return QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()

    # Główna pętla sterująca pobieraniem danych z kolejek i synchronizacją czasową.
    def run(self):
        self.model = YOLO("yolov8n-pose.pt")
        self.model.to(0)
        latest_a = None
        latest_b = None

        while self._is_running:
            try:
                # Próba pobrania najnowszej klatki z kamery laptopa.
                if latest_a is None:
                    latest_a = self.queue_a.get(timeout=0.01)
            except queue.Empty:
                pass

            try:
                # Próba pobrania najnowszej klatki z kamery telefonu.
                if latest_b is None:
                    latest_b = self.queue_b.get(timeout=0.01)
            except queue.Empty:
                pass

            if latest_a is None and latest_b is None:
                continue

            q_img_a = QImage()
            q_img_b = QImage()

            # Sprawdzenie czy klatki pochodzą z tego samego momentu.
            if latest_a is not None and latest_b is not None:
                time_a, frame_a = latest_a
                time_b, frame_b = latest_b
                time_diff = abs(time_a - time_b)

                # Jeśli klatki są zbyt odległe w czasie, usuwamy starszą i czekamy na nowszą.
                if time_diff > self.sync_threshold:
                    if time_a < time_b:
                        latest_a = None
                    else:
                        latest_b = None
                    continue

                res_a = self.model(frame_a, verbose=False)[0].plot()
                res_b = self.model(frame_b, verbose=False)[0].plot()

                # Informacja o opóźnieniu (DEBUG)
                debug_text = f"Sync Delta: {time_diff:.3f}s"
                cv2.putText(res_a, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(res_b, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                q_img_a = self._np_to_qimage(res_a)
                q_img_b = self._np_to_qimage(res_b)

                latest_a = None
                latest_b = None

            # Obsługa sytuacji, gdy dostępna jest tylko jedna kamera (podgląd bez synchronizacji).
            elif latest_a is not None:
                res_a = self.model(latest_a[1], verbose=False)[0].plot()
                q_img_a = self._np_to_qimage(res_a)
                latest_a = None

            elif latest_b is not None:
                res_b = self.model(latest_b[1], verbose=False)[0].plot()
                q_img_b = self._np_to_qimage(res_b)
                latest_b = None

            self.frames_ready.emit(q_img_a, q_img_b)

    def stop(self):
        self._is_running = False
        self.wait()