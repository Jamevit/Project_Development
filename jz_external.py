"""External video preprocessing for an independent J/Z/NEITHER model.

Never reads or writes the existing keypoint.csv or static model.
"""
from pathlib import Path
import hashlib
import json
import time
from collections import Counter, deque
import numpy as np
import cv2

LABELS = ('NEITHER', 'J', 'Z')
FEATURE_VERSION = 'external-jz-xy32-v1'
STEPS = 32


def inventory(root):
    root = Path(root)
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.upper() not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' or len(folder.name) != 1:
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in ('.avi', '.mp4', '.mov', '.mkv'):
                continue
            cap = cv2.VideoCapture(str(path))
            fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps <= 0 or count < 2:
                raise ValueError('Unreadable video: ' + str(path))
            rows.append({'path': path.relative_to(root).as_posix(), 'source_label': folder.name.upper(),
                         'label': folder.name.upper() if folder.name.upper() in ('J', 'Z') else 'NEITHER',
                         'duration': count/fps})
    if not rows or not {'J','Z'}.issubset({r['label'] for r in rows}):
        raise ValueError('Dataset folder must contain J and Z video subfolders.')
    return rows


def preview(path, start=0, end=None):
    """Play decoded video frames in Jupyter, including AVI without browser codecs."""
    from IPython.display import display, Image
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        cap.release()
        raise ValueError('Cannot open video')
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start*fps))
    handle = display(Image(data=cv2.imencode('.jpg', np.zeros((100, 200, 3), dtype=np.uint8))[1].tobytes()), display_id=True)
    index = int(start*fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (end is not None and index/fps > end):
                break
            cv2.putText(frame, f'{index/fps:.2f}s', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
            handle.update(Image(data=cv2.imencode('.jpg', frame)[1].tobytes()))
            time.sleep(1/fps)
            index += 1
    finally:
        cap.release()


class HandReader:
    def __init__(self, task_path=None):
        import mediapipe as mp
        self.mp = mp
        self.timestamp = 0
        self.legacy = hasattr(mp, 'solutions')
        if self.legacy:
            self.detector = mp.solutions.hands.Hands(max_num_hands=2,
                min_detection_confidence=.5, min_tracking_confidence=.5)
        else:
            if task_path is None or not Path(task_path).is_file():
                raise FileNotFoundError('Set TASK_PATH to hand_landmarker.task in the configuration cell.')
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(task_path).read_bytes()),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
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
            self.timestamp += 67
            result = self.detector.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.timestamp)
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


def tip_path_length(points, landmark_id):
    points = np.asarray(points, dtype=np.float32)
    scale = np.median(np.linalg.norm(points[:,9]-points[:,0], axis=1))
    if scale < 5:
        return 0.0
    relative = (points-points[:,0:1])/scale
    return float(np.linalg.norm(np.diff(relative[:,landmark_id], axis=0), axis=1).sum())


class LiveMotionClassifier:
    def __init__(self, model_root='model/jz_external', score_threshold=.8,
                 inference_interval=.1):
        import tensorflow as tf
        models = sorted(Path(model_root).glob('run-*/motion.keras'))
        if not models:
            raise FileNotFoundError('No trained J/Z motion model found under ' + str(model_root))
        self.model_path = models[-1]
        self.model = tf.keras.models.load_model(self.model_path)
        self.model_inference = tf.function(
            lambda features: self.model(features, training=False),
            reduce_retracing=True)
        self.score_threshold = score_threshold
        self.inference_interval = inference_interval
        self.frames = deque(maxlen=48)
        self.predictions = deque(maxlen=5)
        self.last_sample_time = None
        self.last_inference_time = None
        self.label = ''
        self.debug_text = 'Show one hand'
        self.last_sign_motion_time = None
        self.last_high_conf_z_time = None

    def reset(self, status='Show one hand'):
        self.frames.clear()
        self.predictions.clear()
        self.last_sample_time = None
        self.last_inference_time = None
        self.label = ''
        self.debug_text = status
        self.last_sign_motion_time = None
        self.last_high_conf_z_time = None

    def update(self, points, side):
        now = time.monotonic()
        if side not in ('Left', 'Right'):
            self.reset('Hand side not detected')
            return self.label
        if self.frames and (now - self.frames[-1][1] > .5 or self.frames[-1][2] != side):
            self.reset()
        if self.last_sample_time is not None and now - self.last_sample_time < 1 / 15:
            return self.label

        self.frames.append((np.asarray(points, dtype=np.float32), now, side))
        self.last_sample_time = now
        while self.frames and now - self.frames[0][1] > 3.2:
            self.frames.popleft()
        if len(self.frames) < 8:
            self.debug_text = f'Collecting motion {len(self.frames)}/8'
            return self.label
        if (self.last_inference_time is not None and
                now - self.last_inference_time < self.inference_interval):
            if (self.last_sign_motion_time is None or
                        now - self.last_sign_motion_time > 1.5):
                self.label = ''
            return self.label
        self.last_inference_time = now

        sequence, times, _ = zip(*self.frames)
        try:
            features = sequence_features(sequence, times, side)
        except ValueError:
            self.predictions.append(None)
            self.label = ''
            self.debug_text = 'Hand tracking interrupted'
            return self.label

        scores = self.model_inference(features[np.newaxis, ...]).numpy()[0]
        class_id = int(np.argmax(scores))
        prediction = LABELS[class_id] if scores[class_id] >= self.score_threshold else None
        if (scores[2] >= .95 and
                (self.last_high_conf_z_time is None or
                 now - self.last_high_conf_z_time > 3.2)):
            self.last_sign_motion_time = now
            self.last_high_conf_z_time = now
        recent_frames = [frame for frame in self.frames if now - frame[1] <= .4]
        recent_points = [frame[0] for frame in recent_frames]
        recent_pinky_path = tip_path_length(recent_points, 20) if len(recent_points) >= 3 else 0.0
        recent_index_path = tip_path_length(recent_points, 8) if len(recent_points) >= 3 else 0.0
        if prediction == 'J' and not (
                recent_pinky_path >= .7 and
                recent_pinky_path >= 1.4 * recent_index_path):
            prediction = None
        elif prediction == 'Z' and recent_index_path < .15 and scores[2] < .95:
            prediction = None
        self.predictions.append(prediction if prediction in ('J', 'Z') else None)
        counts = Counter(label for label in self.predictions if label is not None)
        if prediction == 'J' or (prediction == 'Z' and recent_index_path >= .15):
            self.last_sign_motion_time = now
        if self.last_sign_motion_time is None or now - self.last_sign_motion_time > 1.5:
            self.label = ''
        else:
            self.label = max(counts, key=counts.get) if counts and max(counts.values()) >= 2 else ''
        self.debug_text = (
            f'J:{scores[1]:.2f} Z:{scores[2]:.2f} '
            f'pinky:{recent_pinky_path:.1f} index:{recent_index_path:.1f} '
            f'result:{self.label or "-"}'
        )
        return self.label


def extract(path, task_path=None, start=0, end=None):
    from collections import Counter
    cap = cv2.VideoCapture(str(path))
    reader = None
    points, times, sides = [], [], []
    total = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            raise ValueError('Unreadable video')
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        end = duration if end is None else end
        if not 0 <= start < end <= duration+.05:
            raise ValueError('Invalid trim times')
        reader = HandReader(task_path)
        first = int(start*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        index, stride = first, max(1,round(fps/15))
        while index/fps < end:
            ok, frame = cap.read()
            if not ok:
                break
            if (index-first)%stride == 0:
                total += 1
                found = reader.read(frame)
                if found is not None:
                    p, side = found
                    points.append(p); times.append(index/fps); sides.append(side)
            index += 1
        if len(points) < 8 or len(points)/max(total,1) < .6:
            raise ValueError('Hand visible in too few frames; review clip/trim')
        side, count = Counter(sides).most_common(1)[0]
        if count/len(sides) < .8:
            raise ValueError('Hand side unstable; review clip')
        x = sequence_features(points,times,side)
        return x, {'detected_frames':len(points), 'sampled_frames':total, 'hand':side,
                   'visible_seconds':times[-1]-times[0], 'start':start, 'end':end}
    finally:
        cap.release()
        if reader:
            reader.close()


def prepare(root, rows, trims, task_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    xs, accepted, rejected, digests = [], [], [], set()
    for i,row in enumerate(rows):
        path = Path(root)/row['path']
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in digests:
                raise ValueError('Exact duplicate video')
            start,end = trims.get(row['path'], (0,None))
            x, info = extract(path,task_path,start,end)
            digests.add(digest)
            xs.append(x)
            accepted.append({**row, **info, 'sha256':digest})
        except (ValueError, OSError) as exc:
            rejected.append({**row,'reason':str(exc)})
        print(f'{i+1}/{len(rows)} processed; {len(accepted)} accepted; {len(rejected)} rejected', flush=True)
    (out/'extraction_report.json').write_text(json.dumps({'accepted':accepted,'rejected':rejected},indent=2),encoding='utf-8')
    if not accepted:
        raise ValueError('No usable clips. Inspect extraction_report.json.')
    np.savez_compressed(out/'sequences.npz', x=np.stack(xs),
                        y=np.array([LABELS.index(r['label']) for r in accepted]),
                        metadata=json.dumps(accepted), feature_version=FEATURE_VERSION)
    return accepted,rejected


def split_by_video(records):
    # Each whole source video belongs to one split, never individual frames.
    rng = np.random.default_rng(42)
    groups = {'train':[], 'validation':[], 'test':[]}
    for source in sorted({r['source_label'] for r in records}):
        ids = np.array([i for i,r in enumerate(records) if r['source_label']==source])
        rng.shuffle(ids)
        if source in ('J','Z') and len(ids)<5:
            raise ValueError(f'Only {len(ids)} usable {source} videos; need at least 5.')
        if len(ids)<3:
            groups['train'].extend(ids.tolist())
            continue
        n = max(1,int(round(len(ids)*.2)))
        groups['test'].extend(ids[:n].tolist())
        groups['validation'].extend(ids[n:2*n].tolist())
        groups['train'].extend(ids[2*n:].tolist())
    for name,ids in groups.items():
        if {records[i]['label'] for i in ids} != set(LABELS):
            raise ValueError(name+' needs usable J, Z and NEITHER examples.')
    return groups


def load_personal_recordings(root):
    root = Path(root)
    xs, ys, records = [], [], []
    for path in sorted(root.rglob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            label = str(data['label'].item()).upper()
            if label not in LABELS:
                raise ValueError('Unknown recording label in ' + str(path))
            features = sequence_features(
                data['points'], data['timestamps'], str(data['hand'].item()))
        xs.append(features)
        ys.append(LABELS.index(label))
        records.append({'path':path.relative_to(root).as_posix(), 'label':label})
    if not records:
        raise ValueError('No usable recordings found under ' + str(root))
    return np.stack(xs), np.asarray(ys, dtype=np.int64), records


def split_personal_recordings(labels):
    rng = np.random.default_rng(42)
    groups = {'train':[], 'validation':[], 'test':[]}
    for label_id in range(len(LABELS)):
        ids = np.flatnonzero(labels == label_id)
        if len(ids) < 5:
            raise ValueError(f'Need at least 5 personal recordings for {LABELS[label_id]}.')
        rng.shuffle(ids)
        test_count = max(1, int(round(len(ids) * .2)))
        validation_count = max(1, int(round(len(ids) * .2)))
        groups['test'].extend(ids[:test_count].tolist())
        groups['validation'].extend(ids[test_count:test_count + validation_count].tolist())
        groups['train'].extend(ids[test_count + validation_count:].tolist())
    return groups


def train(cache, output, epochs=120, recordings=None):
    from datetime import datetime
    import tensorflow as tf
    with np.load(cache, allow_pickle=False) as d:
        if d['feature_version'].item()!=FEATURE_VERSION:
            raise ValueError('Feature version mismatch')
        x,y = d['x'],d['y']
        records = json.loads(d['metadata'].item())
    if x.shape != (len(y),STEPS,42) or len(records)!=len(y) or not np.isfinite(x).all():
        raise ValueError('Invalid feature cache')
    groups = split_by_video(records)
    a,b,c = (groups[k] for k in ('train','validation','test'))
    recording_root = Path(recordings) if recordings is not None else Path(output).resolve().parents[1] / 'jz_recordings'
    if recording_root.is_dir():
        personal_x, personal_y, personal_records = load_personal_recordings(recording_root)
        personal_groups = split_personal_recordings(personal_y)
    else:
        personal_x = np.empty((0,STEPS,42), dtype=np.float32)
        personal_y = np.empty((0,), dtype=np.int64)
        personal_records = []
        personal_groups = {'train':[], 'validation':[], 'test':[]}
    personal_train, personal_validation, personal_test = (
        personal_groups[k] for k in ('train','validation','test'))
    train_x = np.concatenate((x[a], personal_x[personal_train]))
    train_y = np.concatenate((y[a], personal_y[personal_train]))
    validation_x = np.concatenate((x[b], personal_x[personal_validation]))
    validation_y = np.concatenate((y[b], personal_y[personal_validation]))
    test_x = np.concatenate((x[c], personal_x[personal_test]))
    test_y = np.concatenate((y[c], personal_y[personal_test]))
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((STEPS,42)),
        tf.keras.layers.GRU(24),tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(16,activation='relu'),
        tf.keras.layers.Dense(3,activation='softmax')])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    counts = np.bincount(train_y,minlength=3)
    weights = {i:len(train_y)/(3*int(counts[i])) for i in range(3)}
    history = model.fit(train_x,train_y,validation_data=(validation_x,validation_y),epochs=epochs,batch_size=16,
              class_weight=weights, callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor='val_loss',patience=15,restore_best_weights=True)],verbose=2)
    scores = model(test_x,training=False).numpy()
    pred = scores.argmax(axis=1)
    matrix = np.zeros((3,3),dtype=int)
    for actual,guess in zip(test_y,pred):
        matrix[actual,guess]+=1
    recalls = np.diag(matrix)/matrix.sum(axis=1)
    out = Path(output)/datetime.now().strftime('run-%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    model.save(out/'motion.keras')
    report = {'feature_version':FEATURE_VERSION,'labels':LABELS,
              'accuracy':float(np.mean(pred==test_y)), 'balanced_accuracy':float(recalls.mean()),
              'recall_by_class':dict(zip(LABELS,recalls.tolist())),
              'confusion_matrix':matrix.tolist(),'test_count':len(test_y),
              'splits':{k:[records[i] for i in ids] for k,ids in groups.items()},
              'personal_recordings':{'root':str(recording_root),
                  'counts':dict(Counter(r['label'] for r in personal_records)),
                  'splits':{k:[personal_records[i] for i in ids]
                            for k,ids in personal_groups.items()}},
              'source':'SignAlphaSet v3, https://doi.org/10.17632/8fmvr9m98w.3 (CC BY 4.0)',
              'limitation':'Personal recordings are split by clip, not by session. This does not establish signer-independent or continuous webcam accuracy.'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'history.json').write_text(json.dumps(history.history,indent=2),encoding='utf-8')
    return model,history,report,out
