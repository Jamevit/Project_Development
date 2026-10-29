"""SignBridge: one complete two-second recording per LSTM prediction."""
from pathlib import Path
import base64
import binascii
import os
from threading import Lock
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException
from hand_sequences import HandReader, FEATURE_VERSION
from lstm_webcam import load_lstm_model, clip_input, top_predictions

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'model/keypoint_classifier/lstm_classifier.keras'
DEFAULT_ORIGINS = {f'http://{host}:{port}' for host in ('localhost', '127.0.0.1') for port in (5010,5500,5501)}

def decode_frame(value):
    if not isinstance(value, str) or len(value) > 700000:
        raise ValueError('Invalid frame or frame too large.')
    header, sep, encoded = value.partition(',')
    if not sep or header != 'data:image/jpeg;base64':
        raise ValueError('Each frame must be a JPEG data URL.')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError('Invalid image data.')
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[0]*frame.shape[1] > 1000000:
        raise ValueError('Invalid image size.')
    return frame

def create_app(model=None, reader_factory=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024
    if model is None:
        model = load_lstm_model(MODEL_PATH)
    if model.input_shape != (None,1344) or model.output_shape != (None,26):
        raise ValueError('Expected an A-Z LSTM with input (None,1344) and output (None,26).')
    if reader_factory is None:
        reader_factory = lambda: HandReader(ROOT/'hand_landmarker.task')
    lock = Lock()
    allowed = os.environ.get('SIGNBRIDGE_ALLOWED_ORIGINS')
    origins = {x.strip() for x in allowed.split(',')} if allowed else DEFAULT_ORIGINS

    @app.after_request
    def cors(response):
        if request.headers.get('Origin') in origins:
            response.headers['Access-Control-Allow-Origin'] = request.headers['Origin']
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.vary.add('Origin')
        response.headers['Cache-Control'] = 'no-store'
        return response

    website = ROOT.parent.parent / 'SignBridge_Webpage'

    @app.get('/')
    @app.get('/index.html')
    def website_index():
        return send_from_directory(website, 'index.html')

    @app.get('/app.js')
    def website_script():
        return send_from_directory(website, 'app.js')

    @app.get('/styles.css')
    def website_style():
        return send_from_directory(website, 'styles.css')

    @app.get('/health')
    def health():
        return jsonify(status='ready', model='lstm_classifier.keras', mode='sequence',
                       preprocessing=FEATURE_VERSION, seconds=2, classes=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

    @app.post('/predict-sequence')
    def predict_sequence():
        data = request.get_json(silent=True)
        try:
            if not isinstance(data, dict):
                raise ValueError('Send a recording as JSON.')
            frames, times = data.get('frames'), data.get('timestamps')
            if not isinstance(frames, list) or not 15 <= len(frames) <= 60:
                raise ValueError('Record 15 to 60 frames for one sign.')
            t = np.asarray(times, dtype=float)
            if t.shape != (len(frames),) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
                raise ValueError('Invalid recording timestamps.')
            if t[0] < 0 or t[-1] > 2.5 or not 1.6 <= t[-1]-t[0] <= 2.5:
                raise ValueError('Record a complete two-second sign.')
            selected = data.get('hand')
            if selected not in ('Left', 'Right'):
                raise ValueError('Choose the hand used for this recording.')
            points, tracked_times, sides = [], [], []
            shape = None
            # A fresh detector for each clip prevents tracking state leaking between users.
            with lock:
                reader = reader_factory()
                try:
                    for image, timestamp in zip(frames, t):
                        frame = decode_frame(image)
                        if shape is not None and frame.shape != shape:
                            raise ValueError('Camera resolution changed. Record again.')
                        shape = frame.shape
                        found = reader.read(frame)
                        if found is not None:
                            points.append(found[0]); tracked_times.append(timestamp); sides.append(found[1])
                    if sides and any(side != selected for side in sides):
                        raise ValueError('Use your selected hand throughout, or change the hand selection.')
                    features = clip_input(points, tracked_times, sides, len(frames), seconds=2)
                    scores = np.asarray(model(features, training=False))[0]
                    ranking = top_predictions(scores)
                finally:
                    reader.close()
            letter, confidence = ranking[0]
            return jsonify(status='ok', letter=letter, confidence=confidence, hand=sides[0],
                           top_predictions=[{'letter':a,'confidence':b} for a,b in ranking],
                           tracked_frames=len(points), total_frames=len(frames),
                           duration=float(t[-1]-t[0]), model='lstm_classifier.keras')
        except (ValueError, TypeError) as error:
            return jsonify(error=str(error).replace('SPACE to retry.', 'Press Record sign to retry.')), 400
        except Exception:
            app.logger.exception('LSTM prediction failed')
            return jsonify(error='Prediction failed. Check the Python API window.'), 500

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code
    return app

if __name__ == '__main__':
    api = create_app()
    print('SignBridge LSTM API ready on http://127.0.0.1:5011', flush=True)
    api.run(host=os.environ.get('SIGNBRIDGE_HOST','127.0.0.1'),
            port=int(os.environ.get('SIGNBRIDGE_PORT','5011')), debug=False)
