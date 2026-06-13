from .exercise_config import EXERCISE_CONFIG

STATE_IDLE = "idle"
STATE_RECORDING_ATTACK = "recording_attack"
STATE_FINISHED = "finished"

DEFAULT_COOLDOWN_SEC = 0.6
MIN_VALID_FRAMES = 5
MIN_TOTAL_MOTION = 0.08

START_CONFIRMATION_FRAMES = 2
MOTION_DEADZONE = 0.02

DEFAULT_POST_PEAK_FRAMES = 10
DEFAULT_POST_PEAK_DROP_RATIO = 0.30
DEFAULT_POST_PEAK_MIN_DURATION_SEC = 0.75

LEG_EXERCISES = ("step_only", "full")

DEFAULT_MIN_LEG_DISPLACEMENT = 0.10
DEFAULT_RETURN_TO_START_THRESHOLD = 0.06
DEFAULT_RETURN_TO_START_FRAMES = 3


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
		self.frames_after_peak = 0
		self.has_clear_motion = False

		self.start_front_ankle_displacement = None
		self.max_front_ankle_displacement_delta = 0.0
		self.return_to_start_count = 0

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

		return av + 0.5 * adc

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

		if signal >= self.peak_motion_signal:
			self.peak_motion_signal = signal
			self.frames_after_peak = 0
		else:
			self.frames_after_peak += 1

		if self.state == STATE_IDLE:
			if signal >= self.segmenter_config["motion_start_threshold"]:
				self.start_confirmation_count += 1
			else:
				self.start_confirmation_count = 0

			if self.start_confirmation_count >= START_CONFIRMATION_FRAMES:
				self.state = STATE_RECORDING_ATTACK
				self.start_time = timestamp
				self.has_clear_motion = True
				self.start_front_ankle_displacement = self._get_front_ankle_displacement(side_features)
				self.records.append(self._record(timestamp, side_features, front_features, signal))
				return "started"

			return None

		self.records.append(self._record(timestamp, side_features, front_features, signal))

		duration = timestamp - self.start_time

		if duration >= self.segmenter_config["max_duration_sec"]:
			return self._finish(timestamp, "timeout")

		if signal >= self.segmenter_config["motion_start_threshold"]:
			self.has_clear_motion = True

		if signal <= self.segmenter_config["stillness_threshold"]:
			self.stillness_count += 1
		else:
			self.stillness_count = 0

		if duration < self.segmenter_config["min_duration_sec"]:
			return None

		if self.exercise_type in LEG_EXERCISES:
			return self._update_leg_exercise(timestamp, side_features)

		return self._update_arms_only(timestamp, signal, duration)

	def _update_leg_exercise(self, timestamp, side_features):
		if not self.has_clear_motion:
			return None

		if self._has_returned_to_start(side_features):
			return self._finish(timestamp, "return_to_start")

		return None

	def _has_returned_to_start(self, side_features):
		current_displacement = self._get_front_ankle_displacement(side_features)

		if current_displacement is None:
			self.return_to_start_count = 0
			return False

		if self.start_front_ankle_displacement is None:
			self.start_front_ankle_displacement = current_displacement
			return False

		displacement_delta = abs(current_displacement - self.start_front_ankle_displacement)

		if displacement_delta > self.max_front_ankle_displacement_delta:
			self.max_front_ankle_displacement_delta = displacement_delta

		min_leg_displacement = self.segmenter_config.get(
			"min_peak_displacement",
			DEFAULT_MIN_LEG_DISPLACEMENT,
		)
		return_to_start_threshold = self.segmenter_config.get(
			"return_displacement_threshold",
			DEFAULT_RETURN_TO_START_THRESHOLD,
		)
		return_to_start_frames = self.segmenter_config.get(
			"return_zone_frames",
			DEFAULT_RETURN_TO_START_FRAMES,
		)

		if self.max_front_ankle_displacement_delta < min_leg_displacement:
			self.return_to_start_count = 0
			return False

		if displacement_delta <= return_to_start_threshold:
			self.return_to_start_count += 1
		else:
			self.return_to_start_count = 0

		return self.return_to_start_count >= return_to_start_frames

	def _get_front_ankle_displacement(self, side_features):
		value = side_features.get("front_ankle_displacement")

		if value is None:
			return None

		return float(value)

	def _update_arms_only(self, timestamp, signal, duration):
		if self.stillness_count >= self.segmenter_config["stillness_frames"]:
			return self._finish(timestamp, "stillness")

		post_peak_enabled = self.segmenter_config.get("post_peak_enabled", True)
		post_peak_frames = self.segmenter_config.get("post_peak_frames", DEFAULT_POST_PEAK_FRAMES)
		post_peak_drop_ratio = self.segmenter_config.get("post_peak_drop_ratio", DEFAULT_POST_PEAK_DROP_RATIO)
		post_peak_min_duration = self.segmenter_config.get(
			"post_peak_min_duration_sec",
			DEFAULT_POST_PEAK_MIN_DURATION_SEC,
		)

		has_clear_peak = self.peak_motion_signal >= self.segmenter_config["motion_start_threshold"]
		has_enough_frames_after_peak = self.frames_after_peak >= post_peak_frames
		has_dropped_after_peak = signal <= self.peak_motion_signal * post_peak_drop_ratio
		has_enough_duration_for_post_peak = duration >= post_peak_min_duration

		if (
				post_peak_enabled
				and has_clear_peak
				and has_enough_frames_after_peak
				and has_dropped_after_peak
				and has_enough_duration_for_post_peak
		):
			return self._finish(timestamp, "post_peak")

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
		return {
			"timestamp": timestamp,
			"side": side_features,
			"front": front_features,
			"motion_signal": signal,
		}

	def _is_valid_repetition(self):
		if len(self.records) < MIN_VALID_FRAMES:
			return False

		total_motion = sum(record.get("motion_signal", 0.0) for record in self.records)

		if total_motion < MIN_TOTAL_MOTION:
			return False

		if self.exercise_type in LEG_EXERCISES:
			min_leg_displacement = self.segmenter_config.get(
				"min_peak_displacement",
				DEFAULT_MIN_LEG_DISPLACEMENT,
			)

			if self.max_front_ankle_displacement_delta < min_leg_displacement:
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
