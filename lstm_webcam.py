"""Test the saved A-Z LSTM on complete webcam signs; no training or file writes."""
from pathlib import Path
import argparse
import json
import time
import zipfile
import numpy as np
import cv2
from hand_sequences import HandReader, sequence_features

ROOT = Path(__file__).resolve().parent
LABELS = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def load_lstm_model(model_path):
    """Load trained weights across Keras 3 initializer-config versions."""
    import tensorflow as tf
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError as error:
        if "unexpected keyword argument 'input_axes'" not in str(error):
            raise
        # Keras 3.15 writes optional initializer axes that Keras 3.7 cannot read.
        # Remove only unset defaults; non-default axes must not be reinterpreted.
        with zipfile.ZipFile(model_path) as archive:
            config = json.loads(archive.read('config.json'))
        def remove_unset_axes(node):
            if isinstance(node, dict):
                if node.get('module') == 'keras.layers' and node.get('class_name') == 'Dense':
                    settings = node.get('config', {})
                    if 'quantization_config' in settings:
                        if settings['quantization_config'] is not None:
                            raise ValueError('Quantized models require their training Keras version.') from error
                        settings.pop('quantization_config')
                if node.get('module') == 'keras.initializers':
                    settings = node.get('config', {})
                    for name in ('input_axes', 'output_axes'):
                        if name in settings:
                            if settings[name] is not None:
                                raise ValueError('This model uses custom initializer axes; load it with its training Keras version.') from error
                            settings.pop(name)
                for value in node.values():
                    remove_unset_axes(value)
            elif isinstance(node, list):
                for value in node:
                    remove_unset_axes(value)
        remove_unset_axes(config)
        config.pop('compile_config', None)
        model = tf.keras.models.model_from_json(json.dumps(config))
        # Read the original weights directly; the saved file is never rewritten.
        model.load_weights(model_path)
        return model

def clip_input(points, timestamps, sides, attempts, seconds=2.0):
    if not sides or len(set(sides)) != 1:
        raise ValueError('Keep the same single hand visible. SPACE to retry.')
    if len(points) < 15 or len(points) / max(attempts, 1) < .85:
        raise ValueError('Hand tracking incomplete. SPACE to retry.')
    if timestamps[-1] - timestamps[0] < .8 * seconds:
        raise ValueError('Keep your hand visible throughout GO. SPACE to retry.')
    # Identical normalization and 32-step resampling to the notebook loader.
    return sequence_features(points, timestamps, sides[0]).reshape(1, 32 * 42)

def top_predictions(scores):
    scores = np.asarray(scores)
    if scores.shape != (26,) or not np.isfinite(scores).all():
        raise ValueError('The model must return 26 finite A-Z scores.')
    return [(LABELS[i], float(scores[i])) for i in np.argsort(-scores)[:3]]

def draw_hand(frame, found):
    view = cv2.resize(cv2.flip(frame, 1), (800, 600))
    if found is not None:
        for px, py in found[0]:
            cv2.circle(view, (int(px*800/frame.shape[1]), int(py*600/frame.shape[0])),
                       3, (100, 255, 100), -1)
    return view

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=2.0)
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 5:
        parser.error('--seconds must be between 1 and 5')
    model_path = ROOT / 'model/keypoint_classifier/lstm_classifier.keras'
    if not model_path.is_file():
        raise FileNotFoundError('Saved LSTM missing: ' + str(model_path))
    import tensorflow as tf
    model = load_lstm_model(model_path)
    if model.input_shape != (None, 1344) or model.output_shape != (None, 26):
        raise ValueError(f'Expected input (None, 1344), output (None, 26); got {model.input_shape}, {model.output_shape}')
    infer = tf.function(lambda x: model(x, training=False))
    infer(np.zeros((1,1344), dtype=np.float32)).numpy()
    print('Loaded LSTM:', model_path, flush=True)
    print('SPACE: one-second countdown, then perform one complete sign. R: replay. ESC: exit.', flush=True)
    cap, reader = cv2.VideoCapture(args.device), None
    state, status = 'idle', 'SPACE: start a new sign capture'
    ranking, capture_info = [], ''
    points, timestamps, sides, frames, frame_times = [], [], [], [], []
    started, next_sample, replay_start, attempts = 0.0, 0.0, 0.0, 0
    found = None
    try:
        if not cap.isOpened():
            raise RuntimeError('Camera unavailable. Close other camera apps and retry.')
        reader = HandReader(ROOT/'hand_landmarker.task')
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera stopped delivering frames.')
            now = time.monotonic()
            fresh = now >= next_sample
            if fresh:
                found = reader.read(frame)
                next_sample = now + 1/15
            view = draw_hand(frame, found)
            if state == 'capture':
                elapsed = now-started
                if elapsed < 1:
                    status = f'READY {1-elapsed:.1f}s: prepare your hand'
                elif elapsed < 1+args.seconds:
                    status = f'GO {1+args.seconds-elapsed:.1f}s: perform one sign'
                    if fresh:
                        attempts += 1
                        frames.append(view.copy())
                        frame_times.append(elapsed-1)
                        if found is not None:
                            points.append(found[0].copy())
                            timestamps.append(elapsed-1)
                            sides.append(found[1])
                else:
                    try:
                        inputs = clip_input(points,timestamps,sides,attempts,args.seconds)
                        ranking = top_predictions(infer(inputs).numpy()[0])
                        status = 'LAST CAPTURE: top guesses below | SPACE: new capture'
                        capture_info = f'{len(points)}/{attempts} tracked frames | {timestamps[-1]-timestamps[0]:.2f}s | {sides[0]} hand'
                        print('Last capture:',ranking,'|',capture_info,flush=True)
                    except ValueError as error:
                        status = str(error)
                        ranking = []
                        capture_info = f'{len(points)}/{attempts} tracked frames; no valid prediction'
                    state = 'result'
            if state == 'replay' and frames:
                offset = (now-replay_start) % args.seconds
                index = int(np.clip(np.searchsorted(frame_times,offset,side='right')-1,0,len(frames)-1))
                view = frames[index]
            panel = np.zeros((790,800,3),dtype=np.uint8)
            panel[190:] = view
            rows = [('LSTM A-Z | SPACE: capture | R: replay/live | ESC: exit',(255,255,255)),
                    (status,(100,255,255)),
                    ('  |  '.join(f'{i+1}. {letter}: {score:.1%}' for i,(letter,score) in enumerate(ranking)) if ranking else 'No prediction for this capture yet.',(100,255,100)),
                    (capture_info,(220,220,220)),
                    ('Scores are model confidence, not measured accuracy.',(190,190,190)),
                    ('REPLAY: previous captured sign' if state=='replay' else 'LIVE CAMERA | results update only after a new capture',(220,220,220))]
            for i,(text,color) in enumerate(rows):
                cv2.putText(panel,text,(12,26+29*i),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
            cv2.imshow('SignBridge LSTM webcam test',panel)
            key=cv2.waitKey(1)&255
            if key==27: break
            if state=='capture': continue
            if key==32:
                points,timestamps,sides,frames,frame_times=[],[],[],[],[]
                ranking,capture_info,attempts=[], '', 0
                started,state=time.monotonic(),'capture'
            elif key in (ord('r'),ord('R')) and frames:
                if state=='replay': state='result'
                else: state,replay_start='replay',time.monotonic()
    finally:
        if reader is not None: reader.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
