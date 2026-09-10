"""Serve the ASL-HG model using exactly its training-time hand preprocessing."""
import base64
import binascii
import hashlib
import io
import json
from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from preprocessing import Preprocessor

ROOT = Path(__file__).resolve().parent
MODEL_NAME = 'aslhg_candidate.keras'
MAX_BODY = 2 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 4_000_000


def create_app(model=None, processor=None):
    app = Flask(__name__, static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = MAX_BODY
    labels = (ROOT/'class_names.txt').read_text(encoding='utf-8-sig').splitlines()
    if labels != list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        raise ValueError('The model requires its matching ordered A-Z class_names.txt.')
    spec = json.loads((ROOT/'preprocessing.json').read_text())
    if hashlib.sha256((ROOT/'hand_landmarker.task').read_bytes()).hexdigest() != spec['task_sha256']:
        raise ValueError('Hand detector does not match the training configuration.')
    if model is None:
        model = tf.keras.models.load_model(ROOT/MODEL_NAME, compile=False)
    if processor is None:
        processor = Preprocessor(ROOT/'hand_landmarker.task')
    if tuple(model.input_shape[1:]) != (224,224,3) or model.output_shape[-1] != 26:
        raise ValueError('Unexpected model input/output shape.')
    lock = Lock()
    app.extensions['hand_processor'] = processor

    @app.after_request
    def headers(response):
        origin = request.headers.get('Origin')
        allowed = {f'http://{host}:{port}' for host in ('127.0.0.1','localhost') for port in (5500,5501)}
        if origin in allowed:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.vary.add('Origin')
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/')
    def home():
        return send_from_directory(ROOT,'index.html')

    @app.get('/<name>')
    def asset(name):
        if name not in {'index.html','styles.css','app.js'}:
            return jsonify(error='Not found'),404
        return send_from_directory(ROOT,name)

    @app.get('/health')
    def health():
        return jsonify(status='ready',model=MODEL_NAME,preprocessing='raw-hand-v1',classes=labels)

    @app.post('/predict')
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data,dict) or not isinstance(data.get('image'),str):
            return jsonify(error='Send an image string in a JSON object.'),400
        encoded = data['image']
        header, separator, encoded = encoded.partition(',')
        if not separator or header not in {'data:image/jpeg;base64','data:image/png;base64'}:
            return jsonify(error='Send a JPEG or PNG data URL.'),400
        try:
            raw = base64.b64decode(encoded,validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {'JPEG','PNG'} or image.width*image.height > 4_000_000:
                    raise ValueError('Invalid image size or format')
                image.verify()
        except (ValueError,binascii.Error,OSError,UnidentifiedImageError,Image.DecompressionBombError):
            return jsonify(error='The image is invalid or too large.'),400
        try:
            with lock:
                try:
                    crop = processor.process(raw)
                except ValueError:
                    return jsonify(status='no_hand',letter=None,confidence=0,
                                   message='Show exactly one hand, fully inside the camera view.')
                # Same RGB PNG crop pixels as training. Scaling is inside the saved model.
                pixels = np.asarray(crop,dtype=np.float32)[None,...]
                scores = np.asarray(model(pixels,training=False))[0]
            if not np.isfinite(scores).all():
                raise ValueError('Invalid prediction')
            best = int(scores.argmax())
            preview = io.BytesIO()
            crop.save(preview,format='PNG')
            return jsonify(status='ok',letter=labels[best],confidence=float(scores[best]),
                crop='data:image/png;base64,'+base64.b64encode(preview.getvalue()).decode())
        except Exception:
            app.logger.exception('Prediction error')
            return jsonify(error='Prediction failed. Check the server window.'),500

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description),error.code

    return app


if __name__=='__main__':
    app = create_app()
    print('ASL-HG model ready. Open http://127.0.0.1:5010',flush=True)
    try:
        app.run(host='127.0.0.1',port=5010,debug=False)
    finally:
        app.extensions['hand_processor'].close()
