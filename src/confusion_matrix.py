"""Recreate the supplied confusion matrix from its aggregate counts.

These 50 observations are historical project results, not a newly validated
hold-out evaluation. No face images or embeddings are included here.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

LABELS = ["Akayed", "Anik", "Mehedee", "suaib"]
MATRIX = np.array([[2, 0, 0, 0], [0, 13, 0, 0], [0, 0, 15, 0], [0, 1, 0, 19]])


def main():
    display = ConfusionMatrixDisplay(confusion_matrix=MATRIX, display_labels=LABELS)
    display.plot(cmap="Blues", values_format="d", colorbar=True)
    plt.title("Confusion Matrix - Smart Attendance System")
    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/confusion_matrix.png", dpi=220, bbox_inches="tight")
    print(f"Matrix accuracy: {np.trace(MATRIX) / MATRIX.sum():.2%} (49/50)")


if __name__ == "__main__":
    main()
