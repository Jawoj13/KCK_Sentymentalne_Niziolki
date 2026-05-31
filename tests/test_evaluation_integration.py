import unittest

from backend import EvaluationController
from evaluation.features import extract_sequence_features, FeatureStreamExtractor
from evaluation.scoring import evaluate_repetition
from evaluation.segmenter import RepetitionSegmenter


def make_base_keypoints():
	p = [[0.0, 0.0, 0.95] for _ in range(17)]
	p[5] = [100, 100, 0.95];
	p[6] = [150, 100, 0.95];
	p[11] = [100, 250, 0.95];
	p[12] = [150, 250, 0.95]
	p[13] = [120, 330, 0.95];
	p[14] = [185, 330, 0.95];
	p[15] = [100, 400, 0.95];
	p[16] = [160, 400, 0.95]
	p[7] = [115, 115, 0.95];
	p[8] = [145, 115, 0.95];
	p[9] = [125, 130, 0.95];
	p[10] = [155, 130, 0.95]
	return p


def make_arms_frame(wrist_x, timestamp=0.0):
	p = make_base_keypoints();
	p[7] = [(100 + wrist_x) / 2, 100, 0.95];
	p[9] = [wrist_x, 100, 0.95]
	return {"timestamp": timestamp, "keypoints": p}


def make_step_frame(ankle_x, timestamp=0.0):
	p = make_base_keypoints();
	p[15] = [ankle_x, 400, 0.95];
	p[13] = [(100 + ankle_x) / 2, 330, 0.95]
	return {"timestamp": timestamp, "keypoints": p}


def run_segmenter(exercise_type, side_features, front_features=None):
	s = RepetitionSegmenter(exercise_type, dominant_side="left");
	event = None
	for i, item in enumerate(side_features):
		event = s.update(item["timestamp"], item,
		                 front_features[i] if front_features and i < len(front_features) else None)
	return event, s


class EvaluationIntegrationTests(unittest.TestCase):
	def test_arms_only_end_to_end_scores_good_extension(self):
		frames = [make_arms_frame(x, i * 0.1) for i, x in
		          enumerate([125, 145, 165, 185, 205, 205, 205, 205, 205, 205, 205, 205])]
		side = extract_sequence_features(frames, "side", "left");
		event, seg = run_segmenter("arms_only", side)
		self.assertEqual(event, "finished")
		rep = seg.get_repetition();
		result = evaluate_repetition("arms_only", rep["side_sequence"], rep["front_sequence"], "left")
		self.assertGreaterEqual(result["score"], 80.0);
		self.assertIn("front_elbow_extension", result["component_scores"])

	def test_step_only_end_to_end_detects_short_step(self):
		frames = [make_step_frame(x, i * 0.1) for i, x in
		          enumerate([100, 110, 118, 120, 120, 120, 120, 120, 120, 120, 120, 120])]
		side = extract_sequence_features(frames, "side", "left");
		front = extract_sequence_features(frames, "front", "left")
		event, seg = run_segmenter("step_only", side, front);
		self.assertEqual(event, "finished")
		rep = seg.get_repetition();
		result = evaluate_repetition("step_only", rep["side_sequence"], rep["front_sequence"], "left")
		self.assertIn("STEP_TOO_SHORT", {e["code"] for e in result["errors"]});
		self.assertEqual(result["main_feedback"]["camera"], "side")

	def test_backend_evaluation_controller_returns_result(self):
		c = EvaluationController("arms_only", 1, "left");
		result = None
		for i, x in enumerate([125, 145, 165, 185, 205, 205, 205, 205, 205, 205, 205, 205]):
			f = make_arms_frame(x, i * 0.1);
			result = result or c.update(f["timestamp"], f["keypoints"])
		self.assertIsNotNone(result);
		self.assertGreaterEqual(result["score"], 80.0);
		self.assertEqual(c.get_summary()["repetitions_done"], 1)

	def test_feature_stream_extractor_adds_temporal_values(self):
		e = FeatureStreamExtractor("side", "left");
		first = e.update(make_arms_frame(125, 0)["keypoints"], 0);
		second = e.update(make_arms_frame(165, 0.1)["keypoints"], 0.1)
		self.assertEqual(first["front_wrist_velocity"], 0.0);
		self.assertGreater(second["front_wrist_velocity"], 0.0);
		self.assertGreater(second["front_wrist_displacement"], 0.0)


if __name__ == "__main__": unittest.main()
