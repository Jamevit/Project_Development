#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HTTP inference API for the SignBridge browser client."""
import base64
import binascii
import csv
import copy
import itertools
import os
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from model import KeyPointClassifier

LABEL_PATH = 'model/keypoint_classifier/keypoint_classifier_label.csv'
MODEL_NAME = 'keypoint_classifier_asl_dynamic.tflite'
PREPROCESSING = 'xy42-v1'
MAX_BODY = 2 * 1024 * 1024
MAX_PIXELS = 4_000_000
DEFAULT_ORIGINS = {
    f'http://{host}:{port}'
    for host in ('localhost', '127.0.0.1')
    for port in (5010, 5500, 5501)
}


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def create_app(hands=None, classifier=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_BODY
    with open(LABEL_PATH, encoding='utf-8-sig') as file:
        labels = [row[0] for row in csv.reader(file) if row]
    if labels != list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        raise ValueError('Expected ordered A-Z keypoint labels.')

    if hands is None:
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
    if classifier is None:
        classifier = KeyPointClassifier(
            'model/keypoint_classifier/keypoint_classifier_asl_dynamic.tflite')
    lock = Lock()
    app.extensions['signbridge_hands'] = hands

    configured_origins = os.environ.get('SIGNBRIDGE_ALLOWED_ORIGINS')
    allowed_origins = (
        {origin.strip() for origin in configured_origins.split(',') if origin.strip()}
        if configured_origins else DEFAULT_ORIGINS
    )

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get('Origin')
        if origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.vary.add('Origin')
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/health')
    def health():
        return jsonify(status='ready', model=MODEL_NAME,
                       preprocessing=PREPROCESSING, classes=labels)

    @app.post('/predict')
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get('image'), str):
            return jsonify(error='Send an image data URL in a JSON object.'), 400
        header, separator, encoded = data['image'].partition(',')
        if not separator or header not in {
                'data:image/jpeg;base64', 'data:image/png;base64'}:
            return jsonify(error='Send a JPEG or PNG data URL.'), 400
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return jsonify(error='The image data is invalid.'), 400
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[0] * frame.shape[1] > MAX_PIXELS:
            return jsonify(error='The image is invalid or too large.'), 400

        try:
            with lock:
                mirrored = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                detected = results.multi_hand_landmarks or []
                if len(detected) != 1:
                    status = 'multiple_hands' if len(detected) > 1 else 'no_hand'
                    message = ('Show only one hand.' if detected else
                               'Show exactly one hand clearly in the camera view.')
                    return jsonify(status=status, letter=None, confidence=0,
                                   hand=None, message=message)
                landmark_list = calc_landmark_list(mirrored, detected[0])
                features = pre_process_landmark(landmark_list)
                display_points = np.asarray(landmark_list, dtype=np.float32).copy()
                display_points[:, 0] = frame.shape[1] - display_points[:, 0]
                scores = np.asarray(classifier.predict_scores(features), dtype=np.float32)
            if scores.shape != (len(labels),) or not np.isfinite(scores).all():
                raise ValueError('The classifier returned invalid scores.')
            best = int(scores.argmax())
            hand_side = results.multi_handedness[0].classification[0].label
            return jsonify(status='ok', letter=labels[best],
                           confidence=float(scores[best]), hand=hand_side,
                           landmarks=(display_points / [frame.shape[1], frame.shape[0]]).tolist())
        except Exception:
            app.logger.exception('ComVision prediction failed')
            return jsonify(error='Prediction failed. Check the ComVision server.'), 500

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    return app


if __name__ == '__main__':
    api = create_app()
    host = os.environ.get('SIGNBRIDGE_HOST', '127.0.0.1')
    port = int(os.environ.get('SIGNBRIDGE_PORT', '5011'))
    print(f'ComVision API ready on {host}:{port}', flush=True)
    try:
        api.run(host=host, port=port, debug=False)
    finally:
        api.extensions['signbridge_hands'].close()
