import cv2
import face_recognition
import pickle
import os
import csv
from datetime import datetime


ENCODINGS_PATH = "models/face_encodings.pkl"
ATTENDANCE_DIR = "attendance_records"


with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

all_students = set(known_names)
present_students = set()

print("Encodings loaded:", len(known_encodings))


cap = cv2.VideoCapture(0)

print(" Webcam started (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding, box in zip(encodings, boxes):
        matches = face_recognition.compare_faces(
            known_encodings, encoding, tolerance=0.45
        )

        name = "Unknown"

        if True in matches:
            matched_idxs = [i for i, m in enumerate(matches) if m]
            counts = {}

            for i in matched_idxs:
                counts[known_names[i]] = counts.get(known_names[i], 0) + 1

            name = max(counts, key=counts.get)
            present_students.add(name)

        top, right, bottom, left = box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame, name, (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

    cv2.imshow("Webcam Attendance (press q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


os.makedirs(ATTENDANCE_DIR, exist_ok=True)

filename = f"{ATTENDANCE_DIR}/attendance_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Status"])

    for name in sorted(all_students):
        status = "Present" if name in present_students else "Absent"
        writer.writerow([name, status])

print("\n=== FINAL SUMMARY ===")
print("Known persons :", all_students)
print("Present       :", present_students)
print("Absent        :", all_students - present_students)
print(f"Attendance saved to: {filename}")
