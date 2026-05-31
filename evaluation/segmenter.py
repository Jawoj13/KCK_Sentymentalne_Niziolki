from .exercise_config import EXERCISE_CONFIG

STATE_IDLE = "idle"
STATE_RECORDING_ATTACK = "recording_attack"
STATE_FINISHED = "finished"


class RepetitionSegmenter:
	def __init__(self, exercise_type, dominant_side="left", config=None):
		self.exercise_type = exercise_type;
		self.dominant_side = dominant_side;
		self.config = config or EXERCISE_CONFIG;
		self.segmenter_config = self.config[exercise_type]["segmenter"];
		self.reset()

	def reset(self):
		self.state = STATE_IDLE;
		self.records = [];
		self.start_time = None;
		self.stillness_count = 0;
		self.finish_reason = None

	def compute_motion_signal(self, side_features):
		if not side_features: return 0.0
		wv = abs(side_features.get("front_wrist_velocity", 0.0) or 0.0);
		wc = abs(side_features.get("wrist_extension_change", 0.0) or 0.0);
		av = abs(side_features.get("front_ankle_velocity", 0.0) or 0.0)
		if self.exercise_type == "arms_only": return wv + wc
		if self.exercise_type == "step_only": return av
		return 0.5 * wv + 0.5 * av

	def update(self, timestamp, side_features, front_features=None):
		signal = self.compute_motion_signal(side_features)

		if timestamp < self.cooldown_until:
			return None

		if self.state == STATE_IDLE:

			if signal >= self.segmenter_config["motion_start_threshold"]:
				self.state = STATE_RECORDING_ATTACK;
				self.start_time = timestamp;
				self.records.append(self._record(timestamp, side_features, front_features, signal));
				return "started"
			return None
		self.records.append(self._record(timestamp, side_features, front_features, signal))
		self.stillness_count = self.stillness_count + 1 if signal <= self.segmenter_config["stillness_threshold"] else 0
		if timestamp - self.start_time >= self.segmenter_config["max_duration_sec"]:
			self.state = STATE_FINISHED;
			self.finish_reason = "timeout";
			return "finished"
		if timestamp - self.start_time >= self.segmenter_config["min_duration_sec"] and self.stillness_count >= \
				self.segmenter_config["stillness_frames"]:
			self.state = STATE_FINISHED;
			self.finish_reason = "stillness";
			return "finished"
		return None

	def is_finished(self):
		return self.state == STATE_FINISHED

	def get_repetition(self):
		return {"exercise_type": self.exercise_type, "dominant_side": self.dominant_side,
		        "finish_reason": self.finish_reason, "records": list(self.records),
		        "side_sequence": [r["side"] for r in self.records if r.get("side") is not None],
		        "front_sequence": [r["front"] for r in self.records if r.get("front") is not None]}

	def _record(self, timestamp, side_features, front_features, signal):
		return {"timestamp": timestamp, "side": side_features, "front": front_features, "motion_signal": signal}
