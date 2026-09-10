# Run the connected SignBridge prototype

Double-click **START_SIGNBRIDGE.cmd** in this folder. Wait for **Model ready**, then open <http://127.0.0.1:5000>. Choose **Try Camera → Enable Camera** and allow camera access.

Keep the terminal open while practicing. Use **Stop Camera** to release the webcam. Press Ctrl+C in the terminal to stop the server.

The launcher uses the tested Python environment at `%USERPROFILE%\Documents\Codex\sb-env`. No separate Live Server is needed: Flask serves both the website and the prediction API. Internet access is needed to load MediaPipe from its CDN. Your image frames are submitted to the local server on your own computer.

## What changed

- `server.py` loads `best_signbridge_model.keras` and the matching `class_names.txt`, serves the frontend, exposes `/health`, and accepts images at `/predict`.
- `app.js` replaces Open Palm/Fist demo classification with model requests, hand-presence checks, and three repeated high-confidence predictions before displaying a letter.
- Webcam inference uses an unmirrored square hand crop with white connections and red points, approximating the inspected annotated training image. This is not yet a verified match to every training image.
- Camera stop, navigation away, and a hidden tab release the camera and discard stale predictions. A stopped session cannot reopen the camera when a permission request completes late.
- `index.html` labels practice as A–Z recognition and removes the old Open Palm target. Other website sections remain prototype interfaces.

The server uses Pillow for decoding and RGB conversion and TensorFlow's bilinear resize to match the training loader. The saved model already rescales pixels, so the server does not divide by 255 again.

## Verification performed

- Actual saved model loaded with TensorFlow 2.21.0 / Keras 3.15.1 on Python 3.12.
- A sample dataset A image posted over HTTP returned A with approximately 100% model confidence. This is only a connection check, not a new accuracy evaluation.
- Six backend tests cover RGB/pixel scaling, malformed data, payload limits, public routes, restricted Live Server CORS, and label validation.
- JavaScript checks cover prediction smoothing, uncertainty, no overlapping requests, stale-result rejection, and cancellation during camera permission.
- The homepage and updated practice page were checked in a browser.

**Live webcam recognition has not been measured.** Crop/background/annotation differences may still affect results. J and Z motion is not evaluated; the model classifies images. Three repeated predictions and a 70% threshold improve display stability but do not prove the signing is correct.

## Troubleshooting

- If the server says the address is in use, a SignBridge server may already be running. Use its page or stop that server before starting another.
- If the camera is unavailable, check browser permission and close other apps using it.
- If tracking cannot load, check the internet connection and refresh.
- If predictions are poor, keep the hand fully visible against a simple background. Next work should compare captured crops with the dataset and measure accuracy across users.
- If using VS Code Live Server, ports 5500 and 5501 on localhost/127.0.0.1 are allowed. Opening the Flask address is simpler.

## A different computer

Use a short local Python virtual-environment path to avoid Windows path-length installation errors, install `requirements.txt`, and run `python server.py` with that environment's Python. The launcher path is specific to this computer. Keep the trained model and labels beside `server.py`. The original training dataset and training scripts are not required to run inference.
