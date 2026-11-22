"""Evaluate face embeddings on a separate, consented hold-out image set.

IMPORTANT: dataset_eval/ must contain different images from dataset/ used to
create models/face_encodings.pkl. Never evaluate on enrollment images.
"""
from pathlib import Path
import pickle
import face_recognition
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ENCODINGS_PATH = Path("models/face_encodings.pkl")
EVAL_DIR = Path("dataset_eval")


def main():
    if not ENCODINGS_PATH.exists():
        raise SystemExit("Missing encodings: run python src/encode_faces.py with local consented data.")
    if not EVAL_DIR.exists() or not any(EVAL_DIR.iterdir()):
        raise SystemExit("Add separate evaluation images to dataset_eval/<person>/ first.")
    # Pickle can execute arbitrary code. Load only locally generated trusted files.
    with ENCODINGS_PATH.open("rb") as file:
        data = pickle.load(file)
    known_encodings, known_names = data["encodings"], data["names"]
    if not known_encodings:
        raise SystemExit("No known encodings available.")
    y_true, y_pred = [], []
    for person_dir in sorted(EVAL_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        for path in person_dir.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = face_recognition.load_image_file(str(path))
            encodings = face_recognition.face_encodings(image)
            if len(encodings) != 1:
                print(f"Skipped {path}: expected exactly one face")
                continue
            distances = face_recognition.face_distance(known_encodings, encodings[0])
            nearest = int(np.argmin(distances))
            y_true.append(person_dir.name)
            y_pred.append(known_names[nearest])
    if not y_true:
        raise SystemExit("No valid evaluation images were found.")
    labels = sorted(set(y_true) | set(y_pred))
    print("Hold-out samples:", len(y_true))
    print("Accuracy:", f"{accuracy_score(y_true, y_pred):.2%}")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=labels))
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("Note: this is closed-set identification; unknown-person rejection is not tested.")


if __name__ == "__main__":
    main()
