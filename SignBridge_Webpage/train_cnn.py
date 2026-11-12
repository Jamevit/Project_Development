import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================================================
# DATASET PATH
# =========================================================

DATASET_PATH = r"X:\Hand_Proj\Dataset"

print("Dataset exists:", os.path.exists(DATASET_PATH))
print("Dataset path:", DATASET_PATH)


# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42


# =========================================================
# LOAD DATASET
# =========================================================

train_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

validation_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)


# =========================================================
# CLASS NAMES
# =========================================================

class_names = train_dataset.class_names

print("\nClasses:")
print(class_names)

print("\nNumber of classes:")
print(len(class_names))


# Save class names

with open(
    "class_names.txt",
    "w",
    encoding="utf-8"
) as file:

    for name in class_names:
        file.write(name + "\n")


# =========================================================
# PERFORMANCE
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.08),

        layers.RandomZoom(0.1),

        layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08
        )
    ],
    name="data_augmentation"
)


# =========================================================
# CNN MODEL
# =========================================================

model = keras.Sequential(
    [
        layers.Input(
            shape=(IMG_SIZE, IMG_SIZE, 3)
        ),

        data_augmentation,

        layers.Rescaling(
            1.0 / 255
        ),

        # -----------------------------
        # CONVOLUTION BLOCK 1
        # -----------------------------

        layers.Conv2D(
            32,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        layers.MaxPooling2D(),

        # -----------------------------
        # BLOCK 2
        # -----------------------------

        layers.Conv2D(
            64,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        layers.MaxPooling2D(),

        # -----------------------------
        # BLOCK 3
        # -----------------------------

        layers.Conv2D(
            128,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        layers.MaxPooling2D(),

        # -----------------------------
        # BLOCK 4
        # -----------------------------

        layers.Conv2D(
            256,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        layers.MaxPooling2D(),

        # -----------------------------
        # CLASSIFIER
        # -----------------------------

        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.4),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.3),

        layers.Dense(
            len(class_names),
            activation="softmax"
        )
    ]
)


# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# Show model structure

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    ),

    keras.callbacks.ModelCheckpoint(
        "best_signbridge_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]


# =========================================================
# TRAIN
# =========================================================

print("\nStarting training...\n")

history = model.fit(
    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=callbacks
)


# =========================================================
# FINAL EVALUATION
# =========================================================

loss, accuracy = model.evaluate(
    validation_dataset
)


print("\n============================")
print("TRAINING FINISHED")
print("============================")

print(
    f"Validation accuracy: "
    f"{accuracy * 100:.2f}%"
)


# =========================================================
# SAVE MODEL
# =========================================================

model.save(
    "signbridge_asl_model.keras"
)


print("\nModel saved as:")
print("signbridge_asl_model.keras")

print("\nBest model saved as:")
print("best_signbridge_model.keras")

print("\nClass names saved as:")
print("class_names.txt")