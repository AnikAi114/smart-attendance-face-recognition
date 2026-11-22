import cv2
import numpy as np
import joblib

IMAGE_SIZE = 100
MODEL_PATH = "models/face_knn_3datasets.pkl"


TEST_IMAGE_PATH = "example.jpg"  # Supply your own locally consented image



def load_and_preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img.flatten().reshape(1, -1)


def main():
    # model load
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")

    # image load + preprocess
    x = load_and_preprocess(TEST_IMAGE_PATH)

    # predict
    pred_label = model.predict(x)[0]
    print("Predicted label:", pred_label)


if __name__ == "__main__":
    main()
