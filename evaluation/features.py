from .exercise_config import YOLO_KEYPOINTS, MIN_KEYPOINT_CONFIDENCE
from .geometry import angle, center, confidence, distance, has_confidence, point_line_distance, safe_divide


def _point(raw):
	if raw is None: return None
	if hasattr(raw, "tolist"): raw = raw.tolist()
	if len(raw) < 2: return None
	return (float(raw[0]), float(raw[1]), float(raw[2]) if len(raw) >= 3 else 1.0)


def get_keypoint(keypoints, name):
	index = YOLO_KEYPOINTS[name]
	return None if keypoints is None or len(keypoints) <= index else _point(keypoints[index])


def side_names(dominant_side):
	front = dominant_side if dominant_side in ("left", "right") else "left"
	return front, "right" if front == "left" else "left"


def _chain(a, b, c):
	ab = distance(a, b);
	bc = distance(b, c)
	return None if ab is None or bc is None else ab + bc


def _xy(point):
	return None if point is None else (float(point[0]), float(point[1]))


def _safe_angle(a, b, c):
	return angle(a, b, c) if has_confidence(a, b, c, min_confidence=MIN_KEYPOINT_CONFIDENCE) else None


def extract_frame_features(keypoints, camera_view, dominant_side="left", timestamp=None):
	front, rear = side_names(dominant_side)
	p = {name: get_keypoint(keypoints, name) for name in YOLO_KEYPOINTS}
	fs, fe, fw = p[f"{front}_shoulder"], p[f"{front}_elbow"], p[f"{front}_wrist"]
	bs, be, bw = p[f"{rear}_shoulder"], p[f"{rear}_elbow"], p[f"{rear}_wrist"]
	fh, fk, fa = p[f"{front}_hip"], p[f"{front}_knee"], p[f"{front}_ankle"]
	bh, bk, ba = p[f"{rear}_hip"], p[f"{rear}_knee"], p[f"{rear}_ankle"]
	sc = center(p["left_shoulder"], p["right_shoulder"]);
	hc = center(p["left_hip"], p["right_hip"])
	shoulder_width = distance(p["left_shoulder"], p["right_shoulder"])
	torso_length = distance(sc, hc)
	front_leg_length = _chain(fh, fk, fa)
	front_arm_length = _chain(fs, fe, fw)
	body_scale = torso_length or shoulder_width or front_leg_length or 1.0
	torso_lean = angle((hc[0], hc[1] - 100.0, confidence(hc)), hc, sc) if hc and sc else None
	knee_line_raw = point_line_distance(fk, fh, fa)
	front_knee_line_error = safe_divide(knee_line_raw, front_leg_length, None) if knee_line_raw is not None else None
	stance_width = safe_divide(abs(fa[0] - ba[0]), shoulder_width, None) if fa and ba else None
	wrist_extension = safe_divide(distance(fw, fs) or 0.0, front_arm_length, None) if front_arm_length else None
	confs = [confidence(x) for x in [fs, fe, fw, fh, fk, fa, bh, bk, ba] if x is not None]
	feature_confidence = sum(confs) / len(confs) if confs else 0.0
	return {"timestamp": timestamp, "camera_view": camera_view, "dominant_side": dominant_side,
	        "front_elbow_angle": _safe_angle(fs, fe, fw), "rear_elbow_angle": _safe_angle(bs, be, bw),
	        "front_knee_angle": _safe_angle(fh, fk, fa), "back_knee_angle": _safe_angle(bh, bk, ba),
	        "torso_lean_angle": torso_lean, "front_knee_line_error": front_knee_line_error,
	        "stance_width": stance_width, "wrist_extension": wrist_extension, "shoulder_width": shoulder_width,
	        "torso_length": torso_length, "front_leg_length": front_leg_length, "front_arm_length": front_arm_length,
	        "body_scale": body_scale, "front_wrist": _xy(fw), "front_ankle": _xy(fa), "hip_center": _xy(hc),
	        "shoulder_center": _xy(sc), "feature_confidence": feature_confidence,
	        "confidence_ok": feature_confidence >= MIN_KEYPOINT_CONFIDENCE}


def _dt(prev, curr):
	if prev is None: return 1.0
	dt = float(curr.get("timestamp") or 0.0) - float(prev.get("timestamp") or 0.0)
	return dt if dt > 1e-6 else 1.0


def _point_velocity(prev, curr, key, dt, scale):
	if prev is None or prev.get(key) is None or curr.get(key) is None:
		return 0.0

	dx = curr[key][0] - prev[key][0]
	dy = curr[key][1] - prev[key][1]

	velocity = ((dx * dx + dy * dy) ** 0.5) / max(scale, 1e-9) / dt
	return 0.0 if velocity < 0.02 else velocity


def _value_change(prev, curr, key, dt):
	if prev is None or prev.get(key) is None or curr.get(key) is None:
		return 0.0

	change = abs(curr[key] - prev[key]) / dt
	return 0.0 if change < 0.02 else change


def _disp(start, curr, scale):
	if start is None or curr is None: return 0.0
	dx = curr[0] - start[0];
	dy = curr[1] - start[1]
	return ((dx * dx + dy * dy) ** 0.5) / max(scale, 1e-9)


def add_temporal_features(features):
	if not features: return features
	sw, sa, sh = features[0].get("front_wrist"), features[0].get("front_ankle"), features[0].get("hip_center")
	prev = None
	for item in features:
		dt = _dt(prev, item);
		scale = item.get("body_scale") or 1.0
		item["front_wrist_velocity"] = _point_velocity(prev, item, "front_wrist", dt, scale)
		item["front_ankle_velocity"] = _point_velocity(prev, item, "front_ankle", dt, scale)
		item["wrist_extension_change"] = _value_change(prev, item, "wrist_extension", dt)
		item["front_ankle_displacement"] = _disp(sa, item.get("front_ankle"), scale)
		item["front_wrist_displacement"] = _disp(sw, item.get("front_wrist"), scale)
		item["hip_center_displacement"] = _disp(sh, item.get("hip_center"), scale)
		item["front_ankle_displacement_change"] = _value_change(prev, item, "front_ankle_displacement", dt)
		prev = item
	return features


def extract_sequence_features(frames, camera_view, dominant_side="left"):
	result = []
	for i, frame in enumerate(frames):
		keypoints = frame.get("keypoints", frame) if isinstance(frame, dict) else frame
		timestamp = frame.get("timestamp", i) if isinstance(frame, dict) else i
		result.append(extract_frame_features(keypoints, camera_view, dominant_side, timestamp))
	return add_temporal_features(result)


class FeatureStreamExtractor:
	def __init__(self, camera_view, dominant_side="left"):
		self.camera_view = camera_view;
		self.dominant_side = dominant_side;
		self.reset()

	def reset(self):
		self.previous_features = None;
		self.start_wrist = None;
		self.start_ankle = None;
		self.start_hip = None

	def update(self, keypoints, timestamp=None):
		item = extract_frame_features(keypoints, self.camera_view, self.dominant_side, timestamp)
		if self.start_wrist is None: self.start_wrist = item.get("front_wrist")
		if self.start_ankle is None: self.start_ankle = item.get("front_ankle")
		if self.start_hip is None: self.start_hip = item.get("hip_center")
		dt = _dt(self.previous_features, item);
		scale = item.get("body_scale") or 1.0
		item["front_wrist_velocity"] = _point_velocity(self.previous_features, item, "front_wrist", dt, scale)
		item["front_ankle_velocity"] = _point_velocity(self.previous_features, item, "front_ankle", dt, scale)
		item["wrist_extension_change"] = _value_change(self.previous_features, item, "wrist_extension", dt)
		item["front_ankle_displacement"] = _disp(self.start_ankle, item.get("front_ankle"), scale)
		item["front_wrist_displacement"] = _disp(self.start_wrist, item.get("front_wrist"), scale)
		item["hip_center_displacement"] = _disp(self.start_hip, item.get("hip_center"), scale)
		item["front_ankle_displacement_change"] = _value_change(self.previous_features, item,
		                                                        "front_ankle_displacement", dt)
		self.previous_features = item
		return item
