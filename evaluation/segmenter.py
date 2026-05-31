from .exercise_config import EXERCISE_CONFIG

STATE_IDLE = "idle"
STATE_RECORDING_ATTACK = "recording_attack"
STATE_FINISHED = "finished"

DEFAULT_COOLDOWN_SEC = 0.6
MIN_VALID_FRAMES = 5
MIN_TOTAL_MOTION = 0.08
START_CONFIRMATION_FRAMES = 1
MOTION_DEADZONE = 0.02


class RepetitionSegmenter:
	def __init__(self, exercise_type, dominant_side="left", config=None):
		self.exercise_type = exercise_type
		self.dominant_side = dominant_side
		self.config = config or EXERCISE_CONFIG
		self.segmenter_config = self.config[exercise_type]["segmenter"]
		self.cooldown_until = 0.0
		self.reset()

	def reset(self):
		self.state = STATE_IDLE
		self.records = []
		self.start_time = None
		self.stillness_count = 0
		self.finish_reason = None
		self.start_confirmation_count = 0
		self.peak_motion_signal = 0.0

	def compute_motion_signal(self, side_features):
		if not side_features:
			return 0.0

		wv = abs(side_features.get("front_wrist_velocity", 0.0) or 0.0)
		wc = abs(side_features.get("wrist_extension_change", 0.0) or 0.0)
		av = abs(side_features.get("front_ankle_velocity", 0.0) or 0.0)
		adc = abs(side_features.get("front_ankle_displacement_change", 0.0) or 0.0)

		if wv < MOTION_DEADZONE:
			wv = 0.0
		if wc < MOTION_DEADZONE:
			wc = 0.0
		if av < MOTION_DEADZONE:
			av = 0.0
		if adc < MOTION_DEADZONE:
			adc = 0.0

		if self.exercise_type == "arms_only":
			return wv + wc

		if self.exercise_type == "step_only":
			return av + 0.5 * adc

		return 0.5 * wv + 0.5 * av

	def update(self, timestamp, side_features, front_features=None):
		if timestamp is None:
			return None

		if timestamp < self.cooldown_until:
			return None

		if not side_features or not side_features.get("confidence_ok", True):
			if self.state == STATE_RECORDING_ATTACK:
				self.stillness_count += 1
			return None

		signal = self.compute_motion_signal(side_features)
		self.peak_motion_signal = max(self.peak_motion_signal, signal)

		if self.state == STATE_IDLE:
			if signal >= self.segmenter_config["motion_start_threshold"]:
				self.start_confirmation_count += 1
			else:
				self.start_confirmation_count = 0

			if self.start_confirmation_count >= START_CONFIRMATION_FRAMES:
				self.state = STATE_RECORDING_ATTACK
				self.start_time = timestamp
				self.records.append(self._record(timestamp, side_features, front_features, signal))
				return "started"

			return None

		self.records.append(self._record(timestamp, side_features, front_features, signal))

		if signal <= self.segmenter_config["stillness_threshold"]:
			self.stillness_count += 1
		else:
			self.stillness_count = 0

		duration = timestamp - self.start_time

		if duration >= self.segmenter_config["max_duration_sec"]:
			return self._finish(timestamp, "timeout")

		if duration >= self.segmenter_config["min_duration_sec"]:
			if self.stillness_count >= self.segmenter_config["stillness_frames"]:
				return self._finish(timestamp, "stillness")

		return None

	def is_finished(self):
		return self.state == STATE_FINISHED

	def get_repetition(self):
		if self.state != STATE_FINISHED:
			return None

		return {
			"exercise_type": self.exercise_type,
			"dominant_side": self.dominant_side,
			"finish_reason": self.finish_reason,
			"records": list(self.records),
			"side_sequence": [r["side"] for r in self.records if r.get("side") is not None],
			"front_sequence": [r["front"] for r in self.records if r.get("front") is not None],
		}

	def _record(self, timestamp, side_features, front_features, signal):
		return {"timestamp": timestamp, "side": side_features, "front": front_features, "motion_signal": signal}

	def _is_valid_repetition(self):
		if len(self.records) < MIN_VALID_FRAMES:
			return False

		total_motion = sum(record.get("motion_signal", 0.0) for record in self.records)

		if total_motion < MIN_TOTAL_MOTION:
			return False

		return True

	def _finish(self, timestamp, reason):
		if not self._is_valid_repetition():
			self.cooldown_until = timestamp + DEFAULT_COOLDOWN_SEC
			self.reset()
			return None

		self.state = STATE_FINISHED
		self.finish_reason = reason
		self.cooldown_until = timestamp + DEFAULT_COOLDOWN_SEC
		return "finished"
