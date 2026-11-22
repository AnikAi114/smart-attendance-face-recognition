import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import joblib

# ===== CONFIG =====
DATASET_DIRS = [
    "dataset/archive 1",
    "dataset/archive 2",
    "dataset/archive 3",
]

IMAGE_SIZE = 128
MODEL_PATH = "models/face_knn_hog.pkl"
MAX_IMAGES_PER_CLASS = 50

X = []
y = []


def extract_hog(gray_img):
    """Return HOG features from a grayscale image."""
    features = hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features


def load_from_folder(root_dir, prefix):
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            continue

        files = [
            f for f in os.listdir(label_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(files)
        files = files[:MAX_IMAGES_PER_CLASS]

        for f in files:
            img_path = os.path.join(label_path, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            feat = extract_hog(img)
            X.append(feat)
            y.append(f"{prefix}_{label}")


for d in DATASET_DIRS:
    if not os.path.isdir(d):
        print("Warning: folder not found ->", d)
        continue
    prefix = os.path.basename(d)
    print("Loading from:", d)
    load_from_folder(d, prefix)

X = np.array(X)
y = np.array(y)

print("Total images:", len(X))
print("Total classes:", len(set(y)))

if len(X) == 0:
    raise SystemExit("No images found! Check folder structure.")

# === Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# === Evaluation ===
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("HOG + KNN Accuracy:", acc)

# === Model save ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("HOG model saved:", MODEL_PATH)
