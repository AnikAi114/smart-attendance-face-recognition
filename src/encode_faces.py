import face_recognition
import os
import pickle

DATASET_DIR = "dataset"
ENCODINGS_PATH = "models/face_encodings.pkl"

known_encodings = []
known_names = []

print("=== Encoding faces from dataset ===")

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"Processing person: {person_name}")

    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(person_path, img_name)

        try:
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f" No face found: {img_name}")
                continue

            known_encodings.append(encodings[0])
            known_names.append(person_name)

        except Exception as e:
            print(f" Error processing {img_name}: {e}")

print(f"\nTotal faces encoded: {len(known_encodings)}")

data = {
    "encodings": known_encodings,
    "names": known_names
}

os.makedirs("models", exist_ok=True)
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f)

print(f" Face encodings saved to: {ENCODINGS_PATH}")
print(" Encoding completed successfully")



