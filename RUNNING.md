# SignBridge with the ASL-HG model

Double-click `START_SIGNBRIDGE.cmd` in your SignBridge folder. Wait for **ASL-HG model ready**, then open **http://127.0.0.1:5010**. Choose Try Camera, then Enable Camera. Keep the server window open. No separate Live Server is needed.

This uses the existing candidate trained on ASL-HG, not a newly retrained model. Its recorded validation accuracy is 70.05%, and its held-out-participant dataset test accuracy is 67.57%. These are not webcam scores. J/Z motion is not recognized.

The browser sends raw, unmirrored camera frames without drawn landmarks. The Flask backend uses the exact `preprocessing.py` and hand detector from training, then passes RGB 0–255 crops to the saved model, which contains its own normalization. A preview shows the actual crop. Frames are processed locally and are not saved by this application. MediaPipe browser assets require internet access.

The model and frontend are checked through `/health` so an older model is not silently used. Example raw images from participant P9 were sent through the real API: B and C were recognized correctly, while A was incorrectly labeled T. For all three, the returned crop was pixel-identical to the prepared dataset crop. No-hand, invalid-image, and private-file access checks passed. Live webcam accuracy remains unmeasured.

The launcher uses `%USERPROFILE%\Documents\Codex\aslhg-env`, which now includes Flask. Stop any other active webcam tab before testing here. The original camera-only files are backed up before installation.
