"""Use this same raw-image preprocessing for training and future live inference."""
from pathlib import Path
import io
import math
import numpy as np
from PIL import Image, ImageOps

SIZE = 224
MARGIN = 0.12


def crop_hand(image, landmarks):
    image = image.convert('RGB')
    w, h = image.size
    xs = [p.x * w for p in landmarks]
    ys = [p.y * h for p in landmarks]
    margin = max(max(xs)-min(xs), max(ys)-min(ys)) * MARGIN
    box = (max(0, math.floor(min(xs)-margin)), max(0, math.floor(min(ys)-margin)),
           min(w, math.ceil(max(xs)+margin)), min(h, math.ceil(max(ys)+margin)))
    if box[2]-box[0] < 10 or box[3]-box[1] < 10:
        raise ValueError('Hand crop is too small')
    crop = ImageOps.contain(image.crop(box), (SIZE, SIZE), Image.Resampling.BILINEAR)
    result = Image.new('RGB', (SIZE, SIZE), 'white')
    result.paste(crop, ((SIZE-crop.width)//2, (SIZE-crop.height)//2))
    return result


class Preprocessor:
    def __init__(self, task_path):
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(task_path).read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def process(self, image_bytes):
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        # Bound detector cost identically for dataset photos and live frames.
        image.thumbnail((960, 960), Image.Resampling.BILINEAR)
        result = self.detector.detect(self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB, data=np.asarray(image)))
        if len(result.hand_landmarks) != 1:
            raise ValueError(f'Expected one hand; detected {len(result.hand_landmarks)}')
        return crop_hand(image, result.hand_landmarks[0])

    def close(self):
        self.detector.close()
