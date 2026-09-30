import numpy as np


TRACE_STEPS = 64
MIN_TRACE_LENGTH = 90.0

_TEMPLATES = {
    'J': np.array([
        [0.68, 0.10], [0.68, 0.35], [0.68, 0.60], [0.67, 0.73],
        [0.62, 0.83], [0.53, 0.90], [0.42, 0.91], [0.32, 0.87],
        [0.27, 0.79], [0.28, 0.70],
    ], dtype=np.float32),
    'Z': np.array([
        [0.12, 0.18], [0.35, 0.18], [0.65, 0.18], [0.88, 0.18],
        [0.70, 0.38], [0.50, 0.59], [0.30, 0.81], [0.12, 0.81],
        [0.37, 0.81], [0.63, 0.81], [0.88, 0.81],
    ], dtype=np.float32),
}


def _resample(points, count=TRACE_STEPS):
    points = np.asarray(points, dtype=np.float32)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], distances > 1.0))
    points = points[keep]
    if len(points) < 2:
        return None

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] < MIN_TRACE_LENGTH:
        return None

    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack([
        np.interp(targets, cumulative, points[:, axis])
        for axis in range(2)
    ]).astype(np.float32)


def _normalize(points):
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    size = maximum - minimum
    if min(size) < 1.0:
        return None
    scaled = (points - minimum) / size
    return scaled - scaled.mean(axis=0)


def _template_variants(points):
    samples = _resample(points * 300.0)
    normalized = _normalize(samples)
    variants = []
    for path in (normalized, normalized[::-1]):
        variants.append(path)
        mirrored = path.copy()
        mirrored[:, 0] *= -1
        variants.append(mirrored)
    return variants


class TraceLetterClassifier:
    def __init__(self, max_distance=0.24, z_max_distance=0.32,
                 min_margin=0.025):
        self.max_distance = max_distance
        self.z_max_distance = z_max_distance
        self.min_margin = min_margin
        self.templates = {
            label: _template_variants(points)
            for label, points in _TEMPLATES.items()
        }

    def classify(self, points):
        samples = _resample(points)
        if samples is None:
            return None
        normalized = _normalize(samples)
        if normalized is None:
            return None

        distances = {}
        for label, templates in self.templates.items():
            distances[label] = min(
                float(np.mean(np.linalg.norm(normalized - template, axis=1)))
                for template in templates
            )

        ordered = sorted(distances.items(), key=lambda item: item[1])
        distance_limit = self.z_max_distance if ordered[0][0] == 'Z' else self.max_distance
        if ordered[0][1] > distance_limit:
            return None
        if ordered[1][1] - ordered[0][1] < self.min_margin:
            return None
        return ordered[0][0]