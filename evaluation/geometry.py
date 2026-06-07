import math


def clamp(value, minimum=0.0, maximum=100.0):
	return max(minimum, min(maximum, value))


def safe_divide(numerator, denominator, default=0.0):
	return default if denominator is None or abs(denominator) < 1e-9 else numerator / denominator


def confidence(point):
	return 1.0 if point is None or len(point) < 3 else float(point[2])


def has_confidence(*points, min_confidence=0.4):
	return all(point is not None and confidence(point) >= min_confidence for point in points)


def xy(point):
	return float(point[0]), float(point[1])


def distance(a, b):
	if a is None or b is None: return None
	ax, ay = xy(a);
	bx, by = xy(b)
	return math.hypot(ax - bx, ay - by)


def center(a, b):
	if a is None or b is None: return None
	ax, ay = xy(a);
	bx, by = xy(b)
	return ((ax + bx) / 2.0, (ay + by) / 2.0, min(confidence(a), confidence(b)))


def angle(a, b, c):
	if a is None or b is None or c is None: return None
	ax, ay = xy(a);
	bx, by = xy(b);
	cx, cy = xy(c)
	v1 = (ax - bx, ay - by);
	v2 = (cx - bx, cy - by)
	l1 = math.hypot(*v1);
	l2 = math.hypot(*v2)
	if l1 < 1e-9 or l2 < 1e-9: return None
	cos_value = clamp((v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2), -1.0, 1.0)
	return math.degrees(math.acos(cos_value))


def point_line_distance(point, line_a, line_b):
	if point is None or line_a is None or line_b is None: return None
	px, py = xy(point);
	ax, ay = xy(line_a);
	bx, by = xy(line_b)
	den = math.hypot(by - ay, bx - ax)
	if den < 1e-9: return math.hypot(px - ax, py - ay)
	return abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax) / den
