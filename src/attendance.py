import os
import cv2
import numpy as np
from datetime import datetime
from skimage.feature import hog
import joblib

# ===== CONFIG =====
MODEL_PATH = "models/face_knn_hog.pkl"     # HOG + KNN model
TODAY_DIR = "Raw_datasets/today_class"
IMAGE_SIZE = 128
ATT_DIR = "attendance_records"


def extract_hog(gray_img):
    features = hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features


def load_and_predict_today():
    """Return HOG features from a grayscale image."""
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")

    all_students = list(model.classes_)
    present_students = set()
    per_image_results = []

    if not os.path.isdir(TODAY_DIR):
        raise SystemExit(f"Folder not found: {TODAY_DIR}")

    for fname in os.listdir(TODAY_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(TODAY_DIR, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Warning: cannot read", img_path)
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        feat = extract_hog(img).reshape(1, -1)

        pred_label = model.predict(feat)[0]
        present_students.add(pred_label)
        per_image_results.append((fname, pred_label))

    return all_students, present_students, per_image_results


def save_attendance_csv(all_students, present_students):
    os.makedirs(ATT_DIR, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(ATT_DIR, f"attendance_{today_str}.csv")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("student_id,status\n")
        for s in all_students:
            status = "Present" if s in present_students else "Absent"
            f.write(f"{s},{status}\n")

    print("Attendance saved to:", csv_path)


def main():
    all_students, present_students, per_image_results = load_and_predict_today()

    print("\n=== Per-image predictions (today_class) ===")
    for fname, label in per_image_results:
        print(f"{fname} -> {label}")

    print("\n=== Summary ===")
    print("Total known students (classes):", len(all_students))
    print("Present students:", len(present_students))
    print("Absent students:", len(set(all_students) - present_students))

    save_attendance_csv(all_students, present_students)


if __name__ == "__main__":
    main()
