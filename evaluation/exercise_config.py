CAMERA_SIDE = "side"
CAMERA_FRONT = "front"
CAMERA_ANY = "any"
EXERCISE_ARMS_ONLY = "arms_only"
EXERCISE_STEP_ONLY = "step_only"
EXERCISE_FULL = "full"
YOLO_KEYPOINTS = {
	"left_shoulder": 5,
	"right_shoulder": 6,
	"left_elbow": 7,
	"right_elbow": 8,
	"left_wrist": 9,
	"right_wrist": 10,
	"left_hip": 11,
	"right_hip": 12,
	"left_knee": 13,
	"right_knee": 14,
	"left_ankle": 15,
	"right_ankle": 16,
}
MIN_KEYPOINT_CONFIDENCE = 0.40
DEFAULT_DTW_SCORE = 100.0
EXERCISE_CONFIG = {
	"arms_only": {
		"score_weights": {
			"front_elbow_extension": 0.50,
			"no_windup": 0.25,
			"torso_stability": 0.15,
			"dtw_reference": 0.10,
		},
		"thresholds": {
			"front_elbow_good": 165.0,
			"windup_max_ratio": 0.10,
			"torso_lean_variation_max": 15.0,
			"hip_displacement_max_ratio": 0.20,
		},
		"segmenter": {
			"motion_start_threshold": 0.25,
			"stillness_threshold": 0.25,
			"stillness_frames": 4,
			"min_duration_sec": 0.25,
			"max_duration_sec": 3.0,
			"post_peak_enabled": True,
			"post_peak_frames": 8,
			"post_peak_drop_ratio": 0.35,
			"post_peak_min_duration_sec": 0.45,
		},
	},
	"step_only": {
		"score_weights": {
			"step_length": 0.25,
			"front_knee_alignment": 0.25,
			"back_knee_bend": 0.20,
			"torso_posture": 0.15,
			"stance_width": 0.10,
			"dtw_reference": 0.05,
		},
		"thresholds": {
			"step_too_short": 0.15,
			"step_good_min": 0.25,
			"step_good_max": 0.70,
			"front_knee_error_warning": 0.08,
			"front_knee_error_bad": 0.15,
			"back_knee_good_max": 170.0,
			"back_knee_too_straight": 175.0,
			"torso_lean_warning": 20.0,
			"torso_lean_bad": 30.0,
			"stance_width_min_ratio": 0.75,
		},
		"segmenter": {
			"motion_start_threshold": 0.25,
			"stillness_threshold": 0.25,
			"stillness_frames": 6,
			"min_duration_sec": 0.45,
			"max_duration_sec": 5.0,
			"post_peak_enabled": True,
			"post_peak_frames": 10,
			"post_peak_drop_ratio": 0.30,
			"post_peak_min_duration_sec": 0.75,
		},
	},
	"full": {
		"score_weights": {
			"arm_extension": 0.30,
			"step_quality": 0.20,
			"synchronization": 0.25,
			"posture": 0.15,
			"dtw_reference": 0.10,
		},
		"thresholds": {
			"foot_starts_too_early_ms": 100.0,
			"sync_excellent_ms": 150.0,
			"sync_ok_ms": 300.0,
			"sync_bad_ms": 500.0,
			"front_elbow_good": 165.0,
			"front_knee_error_bad": 0.15,
			"back_knee_good_max": 170.0,
			"back_knee_too_straight": 175.0,
			"stance_width_min_ratio": 0.75,
			"torso_lean_warning": 20.0,
			"torso_lean_bad": 30.0,
		},
		"segmenter": {
			"motion_start_threshold": 0.25,
			"stillness_threshold": 0.25,
			"stillness_frames": 6,
			"min_duration_sec": 0.55,
			"max_duration_sec": 6.0,
			"post_peak_enabled": True,
			"post_peak_frames": 14,
			"post_peak_drop_ratio": 0.25,
			"post_peak_min_duration_sec": 1.10,
		},
	},
}
ERROR_PRIORITIES = {
	"arms_only": {
		"ARM_NOT_EXTENDED": 1.5,
		"WINDUP": 1.3,
		"TORSO_LEANS_FORWARD": 0.9,
		"LOW_CONFIDENCE": 2.0,
	},
	"step_only": {
		"KNEE_COLLAPSES_INWARD": 1.6,
		"BACK_LEG_STRAIGHT": 1.4,
		"STANCE_NARROWS": 1.2,
		"STEP_TOO_SHORT": 1.0,
		"TORSO_LEANS_FORWARD": 0.9,
		"LOW_CONFIDENCE": 2.0,
	},
	"full": {
		"FOOT_STARTS_BEFORE_HAND": 1.6,
		"HAND_FOOT_NOT_SYNCED": 1.5,
		"ARM_NOT_EXTENDED": 1.3,
		"UNSTABLE_LEGS": 1.2,
		"TORSO_LEANS_FORWARD": 0.9,
		"LOW_CONFIDENCE": 2.0,
	},
}
ERROR_MESSAGES = {
	"ARM_NOT_EXTENDED": "Wyprostuj bardziej rękę prowadzącą.",
	"WINDUP": "Nie rób zamachu, wystarczy prosty wyprost rąk.",
	"TORSO_LEANS_FORWARD": "Utrzymaj bardziej wyprostowany tułów.",
	"KNEE_COLLAPSES_INWARD": "Pilnuj kolana nogi wykrocznej, nie pozwól mu uciekać do środka.",
	"BACK_LEG_STRAIGHT": "Nie prostuj całkowicie tylnej nogi.",
	"STANCE_NARROWS": "Nie zwężaj pozycji podczas kroku.",
	"STEP_TOO_SHORT": "Zrób wyraźniejszy krok do przodu.",
	"FOOT_STARTS_BEFORE_HAND": "Zacznij ruch od rąk, nie od stóp.",
	"HAND_FOOT_NOT_SYNCED": "Spróbuj zakończyć wyprost ręki i krok w tym samym momencie.",
	"UNSTABLE_LEGS": "Utrzymaj stabilniejszą pozycję nóg na końcu ruchu.",
	"LOW_CONFIDENCE": "Nie udało się wiarygodnie ocenić ruchu, ustaw się lepiej w kadrze.",
}
