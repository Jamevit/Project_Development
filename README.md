# SignBridge

## Run the working A-Z LSTM

From this folder:

```powershell
python app.py
```

The existing `python app.py --lstm` command also works. Press Space and perform the sign during GO.

## Record more A-Z clips

```powershell
python record_jz.py --split train --batch 10
```

Despite the retained filename, this recorder now records the full alphabet into `az_recordings` by default. Select a letter, then press Space once. Clips save automatically; Space stops the batch.

## Train

Open `keypoint_classification.ipynb` and run the cells in order. The notebook reads `az_recordings` only. The current trained model is already saved; there is no need to retrain just to run the app.

## Important files

| File or folder | Purpose |
| --- | --- |
| `az_recordings/` | Current A-Z sequence dataset, including J and Z |
| `keypoint_classification.ipynb` | Main LSTM training notebook and saved results |
| `record_jz.py` | Full alphabet recorder |
| `app.py` | LSTM app entry point |
| `lstm_webcam.py` | Webcam capture and compatible model loading |
| `hand_sequences.py` | Shared hand tracking and sequence preprocessing |
| `hand_landmarker.task` | Hand landmark detection model |
| `model/keypoint_classifier/lstm_classifier.keras` | Active trained LSTM |
| `model/personal_lstm_runs/20261011-224421-8baacc/` | Successful training run, model, metrics, split manifest and confusion matrix |
| `web_api.py` | Existing website API |
| `model/keypoint_classifier/keypoint_classifier.py` | Website classifier loader |
| `model/keypoint_classifier/keypoint_classifier_asl_dynamic.tflite` | Model currently used by the website API |
| `model/keypoint_classifier/keypoint_classifier_label.csv` | Website A-Z label order |
| `model/__init__.py` | Python package import for the website classifier |
| `LICENSE`, `.gitignore`, `.git/` | License and version control |

The website API still uses its static model. It has not yet been switched to the LSTM.

The current LSTM scored 75/78 (96.15%) on held-out clips from the same recording sessions. This is not a measurement of accuracy on unseen users.

The retired static webcam app, J/Z-only datasets/models, external datasets, loose archives and older training scripts have been removed.

Original hand-gesture-recognition project: https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
