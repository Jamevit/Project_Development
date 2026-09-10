import os
import cv2
import csv
import mediapipe as mp

# =========================================================
# CHANGE THIS PATH
# Point it to the Data folder that contains A, B, C ... Z
# =========================================================

DATASET_PATH = r"X:\Hand_Proj\Dataset"

OUTPUT_CSV = "landmarks.csv"

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.2
)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_landmarks(landmarks):

    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    wrist_z = landmarks[0].z

    values = []

    for lm in landmarks:

        x = lm.x - wrist_x
        y = lm.y - wrist_y
        z = lm.z - wrist_z

        values.extend([x, y, z])

    max_value = max(abs(v) for v in values)

    if max_value > 0:
        values = [
            v / max_value
            for v in values
        ]

    return values


# =========================================================
# CSV HEADER
# =========================================================

header = ["label"]

for i in range(21):

    header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


# =========================================================
# DATASET CLASSES
# =========================================================

classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(DATASET_PATH, folder)
    )
])

print("Detected classes:")
print(classes)

print(
    "Number of classes:",
    len(classes)
)


# =========================================================
# EXTRACT LANDMARKS
# =========================================================

total = 0
success = 0
failed = 0


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(header)

    for label in classes:

        folder_path = os.path.join(
            DATASET_PATH,
            label
        )

        images = [
            filename
            for filename in os.listdir(folder_path)
            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            )
        ]

        print(
            f"\nProcessing {label}: "
            f"{len(images)} images"
        )

        for index, filename in enumerate(images):

            total += 1

            image_path = os.path.join(
                folder_path,
                filename
            )

            image = cv2.imread(
                image_path
            )

            if image is None:

                failed += 1
                continue

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(
                rgb
            )

            if results.multi_hand_landmarks:

                landmarks = (
                    results
                    .multi_hand_landmarks[0]
                    .landmark
                )

                normalized = (
                    normalize_landmarks(
                        landmarks
                    )
                )

                writer.writerow(
                    [label] + normalized
                )

                success += 1

            else:

                failed += 1

            if (
                index + 1
            ) % 50 == 0:

                print(
                    f"  {index + 1}"
                    f"/{len(images)}"
                )


hands.close()


print("\n========================")
print("FINISHED")
print("========================")

print("Total images:", total)
print("Successful:", success)
print("Failed:", failed)

print(
    "\nSaved to:",
    OUTPUT_CSV
)