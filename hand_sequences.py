"""Hand tracking and 32-frame XY preprocessing shared by recording, training and testing."""
from pathlib import Path
import numpy as np
import cv2

# Persisted in recordings: retain this identifier for existing dataset compatibility.
FEATURE_VERSION = 'external-jz-xy32-v1'
STEPS = 32

class HandReader:
    def __init__(self, task_path=None, static_image_mode=False):
        import mediapipe as mp
        self.mp = mp
        self.timestamp = 0
        self.static_image_mode = static_image_mode
        self.legacy = hasattr(mp, 'solutions')
        if self.legacy:
            self.detector = mp.solutions.hands.Hands(static_image_mode=static_image_mode,
                max_num_hands=2,
                min_detection_confidence=.5, min_tracking_confidence=.5)
        else:
            if task_path is None or not Path(task_path).is_file():
                raise FileNotFoundError('Set TASK_PATH to hand_landmarker.task in the configuration cell.')
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(task_path).read_bytes()),
                running_mode=(mp.tasks.vision.RunningMode.IMAGE if static_image_mode
                              else mp.tasks.vision.RunningMode.VIDEO), num_hands=2,
                min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
                min_tracking_confidence=.5)
            self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def read(self, frame):
        # Same mirrored image convention as the user's existing app.py.
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        if self.legacy:
            result = self.detector.process(rgb)
            marks = result.multi_hand_landmarks
            if not marks or len(marks) != 1:
                return None
            side = result.multi_handedness[0].classification[0].label
            landmarks = marks[0].landmark
        else:
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            if self.static_image_mode:
                result = self.detector.detect(image)
            else:
                self.timestamp += 67
                result = self.detector.detect_for_video(image, self.timestamp)
            if len(result.hand_landmarks) != 1:
                return None
            side = result.handedness[0][0].category_name
            landmarks = result.hand_landmarks[0]
        h, w = frame.shape[:2]
        return np.array([[l.x*w, l.y*h] for l in landmarks], dtype=np.float32), side

    def close(self):
        self.detector.close()

def sequence_features(points, times, side):
    p, t = np.asarray(points, dtype=np.float32), np.asarray(times, dtype=float)
    if p.ndim != 3 or p.shape[1:] != (21,2) or len(p) < 8 or t.shape != (len(p),):
        raise ValueError('Need at least 8 complete hand frames')
    if not np.isfinite(p).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Invalid coordinates or timestamps')
    if np.max(np.diff(t)) > .5 or t[-1]-t[0] < .4:
        raise ValueError('Hand detection interrupted or sequence too short')
    if side not in ('Left', 'Right'):
        raise ValueError('Unknown hand side')
    scale = np.median(np.linalg.norm(p[:,9]-p[:,0], axis=1))
    if scale < 5:
        raise ValueError('Hand too small')
    p = (p-p[0,0])/scale
    if side == 'Left':
        p[:,:,0] *= -1
    flat = p.reshape(len(p),42)
    grid = np.linspace(t[0],t[-1],STEPS)
    return np.stack([np.interp(grid,t,flat[:,i]) for i in range(42)],axis=1).astype(np.float32)
