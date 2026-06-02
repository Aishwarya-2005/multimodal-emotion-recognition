from face_service import predict_face_emotion
from audio_service import predict_audio_emotion
from fusion_service import fuse_emotions

face = predict_face_emotion("test.jpg")

audio = predict_audio_emotion(
r"C:\Users\saiai\Downloads\archive (6)\Actor_01\03-01-03-01-01-01-01.wav"
)

final_emotion = fuse_emotions(
    face,
    audio
)

print("Face :", face)
print("Audio:", audio)
print("Final:", final_emotion)