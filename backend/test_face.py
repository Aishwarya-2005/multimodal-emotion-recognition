from face_service import predict_face_emotion
emotion, confidence = predict_face_emotion(
    "test.jpg"
)

print("Emotion:", emotion)
print("Confidence:", confidence)