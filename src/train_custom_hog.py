import os
import cv2
import numpy as np
import joblib

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ================= CONFIG =================
DATASET_DIR = "dataset"
IMAGE_SIZE = 128
MODEL_PATH = "models/custom_face_knn_hog.pkl"

#  Only these persons will be trained
PERSON_NAMES = None  # Use all class subfolders from your local dataset
# =========================================


def load_and_preprocess(image_path):
    """Load image and extract HOG features"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features


print("\n=== Loading custom face images ===")

X = []
y = []

for person in (PERSON_NAMES or sorted(os.listdir(DATASET_DIR))):
    person_path = os.path.join(DATASET_DIR, person)

    if not os.path.isdir(person_path):
        print(f" Folder not found: {person}")
        continue

    print(f"Loading images for: {person}")

    for file in os.listdir(person_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(person_path, file)
            feat = load_and_preprocess(img_path)

            if feat is not None:
                X.append(feat)
                y.append(person)

X = np.array(X)
y = np.array(y)

print(f"\nTotal images: {len(X)}")
print(f"Persons: {set(y)}")

if len(X) < 10:
    raise SystemExit(" Not enough images to train")


# ================= TRAIN / TEST =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"\n Custom HOG + KNN Accuracy: {acc:.4f}")

# ================= SAVE MODEL =================
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f" Custom model saved: {MODEL_PATH}")
print(" Training finished successfully")
