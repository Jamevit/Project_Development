import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# LOAD LANDMARK DATA
# =========================================================

DATA_FILE = "landmarks.csv"

data = pd.read_csv(
    DATA_FILE
)

print(
    "Dataset shape:",
    data.shape
)

print("\nSamples per class:")

print(
    data["label"]
    .value_counts()
    .sort_index()
)


# =========================================================
# FEATURES AND LABELS
# =========================================================

X = data.drop(
    "label",
    axis=1
)

y = data["label"]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# =========================================================
# RANDOM FOREST
# =========================================================

model = RandomForestClassifier(

    n_estimators=400,

    max_depth=None,

    min_samples_split=2,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"
)


print(
    "\nTraining model..."
)

model.fit(
    X_train,
    y_train
)


# =========================================================
# TEST MODEL
# =========================================================

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n========================")
print("RESULT")
print("========================")

print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        predictions
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "signbridge_model.pkl"
)

print(
    "\nModel saved:"
)

print(
    "signbridge_model.pkl"
)