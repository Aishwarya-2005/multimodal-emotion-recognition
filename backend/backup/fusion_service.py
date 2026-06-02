def fuse_emotions(face_emotion, audio_emotion):

    if face_emotion == audio_emotion:
        return face_emotion

    priority = {
        "angry": 5,
        "fearful": 5,
        "sad": 4,
        "disgust": 4,
        "surprised": 3,
        "happy": 2,
        "neutral": 1,
        "calm": 1
    }

    if priority.get(face_emotion, 0) >= priority.get(audio_emotion, 0):
        return face_emotion

    return audio_emotion