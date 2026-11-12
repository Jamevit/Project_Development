# SignBridge LSTM website

Double-click START_SIGNBRIDGE.cmd in this folder. Wait for **SignBridge LSTM API ready**, then open http://127.0.0.1:5011. Keep the server window open.

1. Open Alphabet Practice and enable the camera.
2. Choose your left or right hand and a reference letter.
3. Press **Record sign (2 seconds)**. Prepare during the three-second countdown.
4. At GO, perform one complete sign during the two-second recording. Hold static signs; trace the full J/Z movement.
5. Wait for the prediction and top three guesses. Press Record sign for another attempt.

The website loads the same model/keypoint_classifier/lstm_classifier.keras as app.py --lstm in SignBridge_V2 - Copy. Restart the API after exporting a newly trained model. Do not rerun training just to start the website.

The API reuses HandReader, clip_input and sequence_features from the working webcam application: mirrored pixel coordinates, handedness normalization and 32-step resampling. Each recording uses its own tracker. No recordings or model weights are changed or saved by the API.

The API requires one consistently tracked hand, at least 15 tracked frames, 85% tracking coverage and at least 1.6 seconds of tracked motion. Numbers are reference-only. Stopping the camera, changing hand/reference or leaving the page cancels an in-progress attempt.

Recording lasts two seconds to match the training clips, and is resampled to 32 steps. Complete one sign within this period. The saved 96.15% evaluation does not measure this browser workflow; evaluate new recordings separately.

If the connection fails, close old API windows and relaunch. The launcher expects Python 3.11 at %LocalAppData%/Programs/Python/Python311/python.exe. The website and API both use port 5011. Only index.html, app.js and styles.css are served; datasets and model files are not exposed. Other browser origins must be listed in SIGNBRIDGE_ALLOWED_ORIGINS. Use localhost or HTTPS for camera permission.
