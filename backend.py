import math
import queue
import time

import cv2
import torch

try:
	from PyQt5.QtCore import QThread, pyqtSignal
	from PyQt5.QtGui import QImage
except Exception:
	class _Signal:
		def __init__(self, *args, **kwargs):
			self._subscribers = []

		def connect(self, callback):
			self._subscribers.append(callback)

		def emit(self, *args, **kwargs):
			for callback in self._subscribers:
				callback(*args, **kwargs)


	def pyqtSignal(*args, **kwargs):
		return _Signal()


	class QThread:
		def __init__(self, *args, **kwargs):
			pass

		def start(self):
			self.run()

		def run(self):
			pass

		def wait(self, *args, **kwargs):
			pass


	class QImage:
		Format_RGB888 = None

		def __init__(self, *args, **kwargs):
			pass

		def copy(self):
			return self

try:
	from ultralytics import YOLO
except Exception:
	YOLO = None

from evaluation.features import FeatureStreamExtractor
from evaluation.segmenter import RepetitionSegmenter
from evaluation.scoring import evaluate_repetition, choose_main_feedback
from evaluation.exercise_config import CAMERA_SIDE, CAMERA_FRONT


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

		d_alpha = self.alpha(self.d_cutoff, dt)
		edx = d_alpha * dx + (1.0 - d_alpha) * self.dx_prev

		velocity_magnitude = torch.norm(edx[..., :2], dim=-1, keepdim=True)
		cutoff = self.min_cutoff + self.beta * velocity_magnitude

		a = self.alpha(cutoff, dt)

		x_hat = x.clone()
		x_hat[..., :2] = a * x[..., :2] + (1.0 - a) * self.x_prev[..., :2]

		self.x_prev = x_hat.clone()
		self.dx_prev = edx.clone()
		self.t_prev = t

		return x_hat


class EvaluationController:
	def __init__(self, exercise_type, target_repetitions=10, dominant_side="left"):
		self.exercise_type = exercise_type
		self.target_repetitions = target_repetitions
		self.dominant_side = dominant_side
		self.segmenter = RepetitionSegmenter(exercise_type, dominant_side=dominant_side)
		self.side_extractor = FeatureStreamExtractor(CAMERA_SIDE, dominant_side=dominant_side)
		self.front_extractor = FeatureStreamExtractor(CAMERA_FRONT, dominant_side=dominant_side)
		self.results = []
		self.last_side_features = None
		self.last_event = None

	def reset(self):
		self.segmenter.reset()
		self.side_extractor.reset()
		self.front_extractor.reset()
		self.results.clear()

	def update(self, timestamp, side_keypoints, front_keypoints=None):
		side_features = self.side_extractor.update(side_keypoints, timestamp) if side_keypoints is not None else None
		front_features = self.front_extractor.update(front_keypoints,
		                                             timestamp) if front_keypoints is not None else None
		self.last_side_features = side_features

		event = self.segmenter.update(timestamp, side_features, front_features)
		self.last_event = event

		if event != "finished":
			return None
		repetition = self.segmenter.get_repetition()

		if repetition is None:
			return None

		result = evaluate_repetition(
			repetition["exercise_type"],
			repetition["side_sequence"],
			repetition["front_sequence"],
			repetition["dominant_side"],
		)

		result["frame_count"] = len(repetition["side_sequence"])
		result["front_frame_count"] = len(repetition["front_sequence"])
		result["finish_reason"] = repetition.get("finish_reason")

		self.results.append(result)
		self.segmenter.reset()
		self.side_extractor.reset()
		self.front_extractor.reset()
		return result

	def get_summary(self):
		if not self.results:
			return {
				"exercise_type": self.exercise_type,
				"target_repetitions": self.target_repetitions,
				"repetitions_done": 0,
				"average_score": 0.0,
				"main_feedback": None,
				"results": [],
			}
		scores = [result.get("score", 0.0) for result in self.results]
		errors = [error for result in self.results for error in result.get("errors", [])]
		return {
			"exercise_type": self.exercise_type,
			"target_repetitions": self.target_repetitions,
			"repetitions_done": len(self.results),
			"average_score": round(sum(scores) / len(scores), 2),
			"best_score": round(max(scores), 2),
			"worst_score": round(min(scores), 2),
			"main_feedback": choose_main_feedback(errors, self.exercise_type) if errors else None,
			"results": list(self.results),
		}

	def get_debug_text(self):
		if not self.last_side_features:
			return (
				f"exercise_type: {self.exercise_type}\n"
				"No side features detected"
			)

		side_features = self.last_side_features
		motion_signal = self.segmenter.compute_motion_signal(side_features)

		return (
			f"exercise_type: {self.exercise_type}\n"
			f"segmenter_state: {self.segmenter.state}\n"
			f"segmenter_event: {self.last_event}\n"
			f"motion_signal: {motion_signal:.3f}\n"
			f"stillness_frames: {self.segmenter.stillness_count}\n"
			f"recorded_frames: {len(self.segmenter.records)}\n"
			f"feature_confidence: {side_features.get('feature_confidence', 0.0):.2f}\n"
			f"confidence_ok: {side_features.get('confidence_ok')}\n"
			f"front_wrist_velocity: {side_features.get('front_wrist_velocity', 0.0):.3f}\n"
			f"front_ankle_velocity: {side_features.get('front_ankle_velocity', 0.0):.3f}\n"
			f"wrist_extension_change: {side_features.get('wrist_extension_change', 0.0):.3f}\n"
			f"wrist_extension: {(side_features.get('wrist_extension') or 0.0):.3f}\n"
			f"front_ankle_displacement: {side_features.get('front_ankle_displacement', 0.0):.3f}\n"
		)


class CameraWorker(QThread):
	def __init__(self, stream_url, frame_queue):
		super().__init__()
		self.stream_url = stream_url
		self.frame_queue = frame_queue
		self._is_running = True

	def run(self):
		capture = cv2.VideoCapture(self.stream_url)
		is_file_source = isinstance(self.stream_url, str)

		fps = capture.get(cv2.CAP_PROP_FPS)
		if fps is None or fps <= 1:
			fps = 30.0

		frame_delay = 1.0 / fps if is_file_source else 0.0

		while self._is_running:
			frame_start_time = time.time()

			ok, frame = capture.read()

			if not ok:
				if is_file_source:
					capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
					time.sleep(frame_delay)
					continue

				time.sleep(0.02)
				continue

			if self.frame_queue.full():
				try:
					self.frame_queue.get_nowait()
				except queue.Empty:
					pass

			try:
				self.frame_queue.put_nowait((time.time(), frame))
			except queue.Full:
				pass

			if is_file_source:
				elapsed = time.time() - frame_start_time
				sleep_time = frame_delay - elapsed
				if sleep_time > 0:
					time.sleep(sleep_time)

		capture.release()

	def stop(self):
		self._is_running = False
		self.wait()


class SyncInferenceWorker(QThread):
	frames_ready = pyqtSignal(QImage, QImage)
	evaluation_ready = pyqtSignal(object)
	debug_ready = pyqtSignal(str)

	def __init__(self, queue_a, queue_b, exercise_type="arms_only", target_repetitions=10, dominant_side="left",
	             model_path="yolov8n-pose.pt"):
		super().__init__()
		self.queue_a = queue_a
		self.queue_b = queue_b
		self._is_running = True
		self.evaluation = EvaluationController(exercise_type, target_repetitions, dominant_side)
		self.model = YOLO(model_path) if YOLO is not None else None

		self.max_track_age = 10
		self.kpts_history_side = {}
		self.kpts_history_front = {}

		if self.model is not None and torch.cuda.is_available():
			self.model.to("cuda")

	def run(self):
		while self._is_running:
			try:
				timestamp, frame_a = self.queue_a.get(timeout=0.05)
			except queue.Empty:
				continue
			frame_b = None
			try:
				_, frame_b = self.queue_b.get(timeout=0.01)
			except queue.Empty:
				pass
			side_keypoints = self._infer_keypoints(
				frame_a,
				timestamp,
				self.kpts_history_side,
			)

			front_keypoints = None

			if frame_b is not None:
				front_keypoints = self._infer_keypoints(
					frame_b,
					timestamp,
					self.kpts_history_front,
				)

			result = self.evaluation.update(
				timestamp,
				side_keypoints,
				front_keypoints,
			)

			debug_text = self.evaluation.get_debug_text()
			if debug_text:
				self.debug_ready.emit(debug_text)

			if result is not None:
				self.evaluation_ready.emit(result)

			self.frames_ready.emit(self._to_qimage(frame_a), self._to_qimage(frame_b))

	def stop(self):
		self._is_running = False
		self.wait()

	def _infer_keypoints(self, frame, timestamp, history_dict):
		if frame is None or self.model is None:
			return None

		results = self.model.track(frame, persist=True, verbose=False)

		if not results:
			return None

		result = results[0]

		if result.keypoints is None or result.keypoints.data is None or len(result.keypoints.data) == 0:
			return None

		if result.boxes is None or result.boxes.id is None:
			current_kpt = result.keypoints.data[0]
			filter_key = "__single_person__"

			if filter_key not in history_dict:
				history_dict[filter_key] = {
					"filter": OneEuroFilter(min_cutoff=0.5, beta=0.01),
					"age": 0,
				}

			smoothed_kpt = history_dict[filter_key]["filter"](timestamp, current_kpt.clone())
			return smoothed_kpt.detach().cpu().numpy() if hasattr(smoothed_kpt, "detach") else smoothed_kpt

		current_kpts = result.keypoints.data
		track_ids = result.boxes.id.int().cpu().tolist()
		active_ids = set(track_ids)

		keys_to_delete = []

		for hist_id in list(history_dict.keys()):
			if hist_id not in active_ids:
				history_dict[hist_id]["age"] += 1
				if history_dict[hist_id]["age"] > self.max_track_age:
					keys_to_delete.append(hist_id)

		for key in keys_to_delete:
			del history_dict[key]

		smoothed_kpts = []

		for track_id, current_kpts in zip(track_ids, current_kpts):
			if track_id not in history_dict:
				history_dict[track_id] = {
					"filter": OneEuroFilter(min_cutoff=0.5, beta=0.01),
					"age": 0,
				}

			filter_instance = history_dict[track_id]["filter"]
			smoothed_kpt = filter_instance(timestamp, current_kpts.clone())

			history_dict[track_id]["age"] = 0
			smoothed_kpts.append((track_id, smoothed_kpt))

		if not smoothed_kpts:
			return None

		selected_track_id, selected_kpt = t = smoothed_kpts[0]

		return selected_kpt.detach().cpu().numpy() if hasattr(selected_kpt, "detach") else selected_kpt

	def _to_qimage(self, frame):
		if frame is None:
			return QImage()
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		height, width, channels = rgb.shape
		return QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
