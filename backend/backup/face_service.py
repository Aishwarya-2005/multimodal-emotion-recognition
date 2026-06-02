import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/cnn_face.keras")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

emotion_labels = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

def predict_face_emotion(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        1.3,
        5
    )

    if len(faces) == 0:
        return "No Face Found"

    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]

    face = cv2.resize(
        face,
        (48, 48)
    )

    face = face / 255.0

    face = face.reshape(
        1,
        48,
        48,
        1
    )

    prediction = model.predict(face)

    emotion = emotion_labels[
        np.argmax(prediction)
    ]

    return emotion