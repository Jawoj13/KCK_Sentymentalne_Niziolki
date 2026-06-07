from .exercise_config import EXERCISE_CONFIG, ERROR_MESSAGES, ERROR_PRIORITIES, DEFAULT_DTW_SCORE, CAMERA_SIDE, \
	CAMERA_FRONT, CAMERA_ANY
from .geometry import clamp, safe_divide


def choose_main_feedback(errors, exercise_type):
	if not errors: return None
	grouped = {};
	priorities = ERROR_PRIORITIES.get(exercise_type, {})
	for e in errors:
		g = grouped.setdefault(e["code"], {"count": 0, "severity": 0.0, "camera": e.get("camera")});
		g["count"] += 1;
		g["severity"] += e.get("severity", 0.0)
	best = max(grouped,
	           key=lambda code: priorities.get(code, 1.0) * (grouped[code]["severity"] / grouped[code]["count"]) *
	                            grouped[code]["count"])
	score = priorities.get(best, 1.0) * grouped[best]["severity"]
	return {"code": best, "message": ERROR_MESSAGES.get(best, "Popraw technikę wykonania."),
	        "camera": grouped[best]["camera"], "score": score}


def _add(errors, exercise_type, code, severity, camera):
	errors.append({"code": code, "severity": round(clamp(severity, 0.0, 1.0), 3), "camera": camera,
	               "message": ERROR_MESSAGES.get(code, "Popraw technikę wykonania."),
	               "priority": ERROR_PRIORITIES.get(exercise_type, {}).get(code, 1.0)})


def weighted_score(scores, weights):
	return clamp(sum(scores.get(k, 0.0) * w for k, w in weights.items()) / max(sum(weights.values()), 1e-9))


def score_elbow_extension(v):
	if v is None: return 0.0
	if v >= 170: return 100.0
	if v >= 165: return 85.0
	if v >= 155: return 65.0
	if v >= 145: return 40.0
	return 20.0


def _idxmax(seq, key):
	return max(range(len(seq)), key=lambda i: seq[i].get(key) if seq[i].get(key) is not None else -999999)


def _first_motion_time(seq, key, threshold):
	for item in seq:
		value = abs(item.get(key, 0.0) or 0.0)
		if value >= threshold:
			return item.get("timestamp")
	return None


def _filter_confident(seq):
	return [item for item in seq if item.get("confidence_ok", True)]


def _avg_conf(*seqs):
	vals = [x.get("feature_confidence") for s in seqs for x in (s or []) if x.get("feature_confidence") is not None]
	return sum(vals) / len(vals) if vals else 0.0


def _result(ex, components, weights, errors, side, front):
	conf = _avg_conf(side, front)
	if conf < 0.4: _add(errors, ex, "LOW_CONFIDENCE", 1.0, CAMERA_ANY)
	return {"exercise_type": ex, "score": round(weighted_score(components, weights), 2),
	        "component_scores": {k: round(v, 2) for k, v in components.items()}, "errors": errors,
	        "main_feedback": choose_main_feedback(errors, ex), "confidence": round(conf, 3)}


def score_no_windup(seq, end_idx, th):
	start = seq[0].get("front_wrist");
	end = seq[end_idx].get("front_wrist");
	sw = seq[0].get("shoulder_width") or 1.0
	if not start or not end: return 100.0, 0.0
	direction = 1 if end[0] - start[0] >= 0 else -1
	max_back = max(
		[max(0, -direction * ((x.get("front_wrist") or start)[0] - start[0])) for x in seq[:max(1, end_idx)]])
	ratio = max_back / max(sw, 1e-9)
	return clamp(100 * (1 - ratio / (th["windup_max_ratio"] * 2))), ratio


def score_torso_stability(seq, th):
	vals = [x.get("torso_lean_angle") for x in seq if x.get("torso_lean_angle") is not None]
	if not vals: return 100.0, 0.0
	severity = clamp((max(vals) - min(vals)) / th["torso_lean_variation_max"] - 1, 0, 1)
	return clamp(100 * (1 - 0.5 * severity)), severity


def score_step_length(seq, idx, th):
	start = seq[0].get("front_ankle");
	end = seq[idx].get("front_ankle");
	leg = seq[idx].get("front_leg_length") or 1.0
	if not start or not end: return 0.0, 0.0
	length = abs(end[0] - start[0]) / max(leg, 1e-9)
	if th["step_good_min"] <= length <= th["step_good_max"]: return 100.0, length
	if length < th["step_good_min"]: return clamp(
		100 * safe_divide(length - th["step_too_short"], th["step_good_min"] - th["step_too_short"], 0.0)), length
	return clamp(100 * safe_divide(0.95 - length, 0.95 - th["step_good_max"], 0.0)), length


def score_front_knee_alignment(v, th):
	if v is None or v <= th["front_knee_error_warning"]: return 100.0
	if v <= th["front_knee_error_bad"]: return 70.0
	return clamp(70 * (1 - (v - th["front_knee_error_bad"]) / th["front_knee_error_bad"]))


def score_back_knee_bend(v, th):
	if v is None or v <= th["back_knee_good_max"]: return 100.0
	if v <= th["back_knee_too_straight"]: return 75.0
	return 20.0


def score_torso_posture(v, th):
	if v is None or v <= 10: return 100.0
	if v <= th.get("torso_lean_warning", 20): return 85.0
	if v <= th.get("torso_lean_bad", 30): return 60.0
	return 30.0


def evaluate_repetition(exercise_type, side_sequence=None, front_sequence=None, dominant_side="left", config=None):
	config = config or EXERCISE_CONFIG
	side = _filter_confident(side_sequence or [])
	front = _filter_confident(front_sequence or [])

	if len(side) < 5:
		errors = []
		_add(errors, exercise_type, "LOW_CONFIDENCE", 1.0, CAMERA_ANY)
		return _result(exercise_type, {}, {}, errors, side, front)

	th = config[exercise_type]["thresholds"]
	weights = config[exercise_type]["score_weights"]
	errors = []

	if exercise_type == "arms_only":
		idx = _idxmax(side, "wrist_extension");
		end = side[idx]
		elbow = score_elbow_extension(end.get("front_elbow_angle"))
		if (end.get("front_elbow_angle") or 0) < th["front_elbow_good"]: _add(errors, exercise_type, "ARM_NOT_EXTENDED",
		                                                                      1.0, CAMERA_ANY)
		wind, ratio = score_no_windup(side, idx, th)
		if ratio > th["windup_max_ratio"]: _add(errors, exercise_type, "WINDUP", 0.8, CAMERA_SIDE)
		torso, sev = score_torso_stability(side, th)
		if sev > 0: _add(errors, exercise_type, "TORSO_LEANS_FORWARD", sev, CAMERA_SIDE)
		return _result(exercise_type, {"front_elbow_extension": elbow, "no_windup": wind, "torso_stability": torso,
		                               "dtw_reference": DEFAULT_DTW_SCORE}, weights, errors, side, front)
	if exercise_type == "step_only":
		idx = _idxmax(side, "front_ankle_displacement");
		end = side[idx];
		fend = front[min(idx, len(front) - 1)] if front else None;
		fstart = front[0] if front else None
		step_score, step_len = score_step_length(side, idx, th)
		if step_len < th["step_good_min"]: _add(errors, exercise_type, "STEP_TOO_SHORT", 1.0, CAMERA_SIDE)
		knee_score = score_front_knee_alignment(fend.get("front_knee_line_error"), th) if fend else 100.0
		if fend and (fend.get("front_knee_line_error") or 0) > th["front_knee_error_bad"]: _add(errors, exercise_type,
		                                                                                        "KNEE_COLLAPSES_INWARD",
		                                                                                        1.0, CAMERA_FRONT)
		back_score = score_back_knee_bend(end.get("back_knee_angle"), th)
		if (end.get("back_knee_angle") or 0) > th["back_knee_too_straight"]: _add(errors, exercise_type,
		                                                                          "BACK_LEG_STRAIGHT", 0.8, CAMERA_SIDE)
		torso = score_torso_posture(end.get("torso_lean_angle"), th)
		stance = 100.0
		if fstart and fend and fstart.get("stance_width") and fend.get("stance_width") and fend["stance_width"] < th[
			"stance_width_min_ratio"] * fstart["stance_width"]:
			stance = 50.0;
			_add(errors, exercise_type, "STANCE_NARROWS", 0.7, CAMERA_FRONT)
		return _result(exercise_type,
		               {"step_length": step_score, "front_knee_alignment": knee_score, "back_knee_bend": back_score,
		                "torso_posture": torso, "stance_width": stance, "dtw_reference": DEFAULT_DTW_SCORE}, weights,
		               errors, side, front)
	hand = _idxmax(side, "wrist_extension")
	foot = _idxmax(side, "front_ankle_displacement")
	peak = side[hand]

	hand_start_time = _first_motion_time(side, "front_wrist_velocity", 0.05)
	foot_start_time = _first_motion_time(side, "front_ankle_velocity", 0.05)

	if hand_start_time is not None and foot_start_time is not None:
		start_delta_ms = (foot_start_time - hand_start_time) * 1000.0
		if start_delta_ms < -th["foot_starts_too_early_ms"]:
			_add(errors, exercise_type, "FOOT_STARTS_BEFORE_HAND", 1.0, CAMERA_SIDE)

	arm = score_elbow_extension(peak.get("front_elbow_angle"))

	if (peak.get("front_elbow_angle") or 0.0) < th["front_elbow_good"]:
		_add(errors, exercise_type, "ARM_NOT_EXTENDED", 1.0, CAMERA_ANY)

	step, _ = score_step_length(
		side,
		foot,
		{"step_good_min": 0.25, "step_good_max": 0.70, "step_too_short": 0.15}
	)

	hand_time = side[hand].get("timestamp")
	foot_time = side[foot].get("timestamp")

	if hand_time is None or foot_time is None:
		sync_ms = abs(hand - foot) * 100.0
	else:
		sync_ms = abs(hand_time - foot_time) * 1000.0

	if sync_ms <= th["sync_excellent_ms"]:
		sync = 100.0
	elif sync_ms <= th["sync_ok_ms"]:
		sync = 85.0
	elif sync_ms <= th["sync_bad_ms"]:
		sync = 60.0
	else:
		sync = 30.0

	if sync_ms > th["sync_ok_ms"]:
		_add(errors, exercise_type, "HAND_FOOT_NOT_SYNCED", 0.8, CAMERA_SIDE)

	posture = score_torso_posture(side[max(hand, foot)].get("torso_lean_angle"), th)

	return _result(
		exercise_type,
		{
			"arm_extension": arm,
			"step_quality": step,
			"synchronization": sync,
			"posture": posture,
			"dtw_reference": DEFAULT_DTW_SCORE,
		},
		weights,
		errors,
		side,
		front,
	)
