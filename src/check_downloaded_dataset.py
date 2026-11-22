import os
import cv2

DATASET_DIRS = [
    "dataset/archive 1",
    "dataset/archive 2",
    "dataset/archive 3",
]

print("=== Checking downloaded datasets ===")

for d in DATASET_DIRS:
    print(f"\nChecking: {d}")

    if not os.path.isdir(d):
        print(" Folder not found")
        continue

    subfolders = [f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]
    print(" Subfolders:", len(subfolders))

    if not subfolders:
        print("No class/person folders found")
        continue

    # Try to read one image
    sample_folder = os.path.join(d, subfolders[0])
    images = [
        f for f in os.listdir(sample_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if not images:
        print("  No images found")
        continue

    img_path = os.path.join(sample_folder, images[0])
    img = cv2.imread(img_path)

    if img is None:
        print("  Image not readable:", img_path)
    else:
        print("  Image readable:", img_path)
