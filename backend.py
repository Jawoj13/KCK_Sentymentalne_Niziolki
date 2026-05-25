import time
import queue
import cv2
import sys
import math
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from ultralytics import YOLO


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        if self.x_prev is None:
            self.x_prev = x.clone()
            self.dx_prev = torch.zeros_like(x)
            self.t_prev = t
            return x.clone()

        dt = t - self.t_prev
        if dt <= 0.0:
            return x

        dx = (x - self.x_prev) / dt
        edx = self.alpha(self.d_cutoff, dt) * dx + (1.0 - self.alpha(self.d_cutoff, dt)) * self.dx_prev

        velocity_magnitude = torch.norm(edx, dim=-1, keepdim=True)
        cutoff = self.min_cutoff + self.beta * velocity_magnitude

        a = self.alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = edx
        self.t_prev = t

        return x_hat

# Przechwytuje strumienie wideo z kamery.
# Wykorzystuje QThread, aby proces dekodowania klatek nie blokował interfejsu użytkownika
class CameraWorker(QThread):
    def __init__(self, stream_url, frame_queue):
        super().__init__()
        self.stream_url = stream_url  # Adres URL (IP) lub indeks kamery (0)
        self.frame_queue = frame_queue
        self._is_running = True

    def _connect(self):
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
                        try:
                            # Odrzucenie najstarszej klatki (powinno pomóc w zapobieganiu opóźnieniom, wymusza przetwarzanie najnowzsych klatek poprzez odrzucenie starszych)
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait((timestamp, frame))
                        except queue.Empty:
                            pass
                else:
                    break

            if capture:
                capture.release()

    def stop(self):
        self._is_running = False
        self.wait()

# Klasa przetwarzająca, synchronizuje obrazy z dwóch źródeł i wykonuje detekcję YOLO.
class SyncInferenceWorker(QThread):
    frames_ready = pyqtSignal(QImage, QImage)

    def __init__(self, queue_a, queue_b):
        super().__init__()
        self.queue_a = queue_a
        self.queue_b = queue_b
        self._is_running = True
        self.model = None
        self.sync_threshold = 0.05

        self.max_track_age = 10
        self.kpts_history_a = {}
        self.kpts_history_b = {}

    def _process_and_filter(self, model_instance, frame, timestamp, history_dict):
        # track() do dostania id obiektu
        result = model_instance.track(frame, persist=True, verbose=False)[0]

        if result.keypoints is not None and result.boxes is not None and result.boxes.id is not None:
            current_kpts = result.keypoints.data
            track_ids = result.boxes.id.int().cpu().tolist()
            active_ids = set(track_ids)

            keys_to_delete = []
            for hist_id in history_dict.keys():
                if hist_id not in active_ids:
                    history_dict[hist_id]["age"] += 1
                    if history_dict[hist_id]["age"] > self.max_track_age:
                        keys_to_delete.append(hist_id)

            for key in keys_to_delete:
                del history_dict[key]

            smoothed_kpts_list = []
            for track_id, current_kpt in zip(track_ids, current_kpts):
                if track_id not in history_dict:
                    history_dict[track_id] = {
                        "filter": OneEuroFilter(min_cutoff=0.5, beta=0.01),
                        "age": 0
                    }

                filter_instance = history_dict[track_id]["filter"]
                smoothed_kpt = filter_instance(timestamp, current_kpt.clone())
                history_dict[track_id]["age"] = 0
                smoothed_kpts_list.append(smoothed_kpt)

            if smoothed_kpts_list:
                result.keypoints.data = torch.stack(smoothed_kpts_list)

        return result.plot()

    def _np_to_qimage(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = channel * width
        return QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()

    # Główna pętla sterująca pobieraniem danych z kolejek i synchronizacją czasową.
    def run(self):
        self.model_a = YOLO("yolov8n-pose.pt")
        self.model_b = YOLO("yolov8n-pose.pt")
        if sys.platform.startswith('linux'):
            self.model_a.to(0)
            self.model_b.to(0)
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

                res_a = self._process_and_filter(self.model_a, frame_a, time_a, self.kpts_history_a)
                res_b = self._process_and_filter(self.model_b, frame_b, time_b, self.kpts_history_b)

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
                time_a, frame_a = latest_a
                res_a = self._process_and_filter(self.model_a, frame_a, time_a, self.kpts_history_a)
                q_img_a = self._np_to_qimage(res_a)
                latest_a = None

            elif latest_b is not None:
                time_b, frame_b = latest_b
                res_b = self._process_and_filter(self.model_b, frame_b, time_b, self.kpts_history_b)
                q_img_b = self._np_to_qimage(res_b)
                latest_b = None

            self.frames_ready.emit(q_img_a, q_img_b)

    def stop(self):
        self._is_running = False
        self.wait()