import cv2
import numpy as np

import os
from pathlib import Path

from tensorflow.keras.models import load_model

# Load trained CNN model for facial emotion recognition
_BASE_DIR = Path(__file__).resolve().parent
model = load_model(str(_BASE_DIR / "models" / "cnn_face.keras"))

# FER2013 has 7 classes, and your face CNN must output 7 logits in this order.
# IMPORTANT: Keep this consistent with `train_face_cnn.py` (NUM_CLASSES=7) and
# the folder names under `data/face/train` and `data/face/val`.
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# Haar-cascade face detector for robust face localization
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def preprocess_face(gray_face: np.ndarray) -> np.ndarray:
    """Resize, normalize and add channel/batch dimensions."""
    face = cv2.resize(gray_face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face


def predict_face_emotion(image_path: str) -> str:
    """Detect face in the image and predict its emotion."""
    image = cv2.imread(image_path)
    if image is None:
        return "Invalid Image"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces; if multiple, use the largest (most prominent) face
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        # Fallback: use the whole image (keeps current behavior but explicit)
        face_roi = gray
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y : y + h, x : x + w]

    face_input = preprocess_face(face_roi)

    prediction = model.predict(face_input, verbose=0)

    emotion_idx = int(np.argmax(prediction))
    if emotion_idx < 0 or emotion_idx >= len(EMOTION_LABELS):
        return "Unknown"

    return EMOTION_LABELS[emotion_idx]