from audio_service import predict_audio_emotion

emotion, confidence = predict_audio_emotion(
    r"C:\Users\saiai\Downloads\archive (6)\Actor_01\03-01-03-01-01-01-01.wav"
)

print("Emotion:", emotion)
print("Confidence:", confidence)