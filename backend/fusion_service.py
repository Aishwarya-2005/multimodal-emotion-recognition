def _normalize_emotion(label: str | None) -> str | None:
    """
    Normalize label names across datasets/models:
    - RAVDESS uses: calm, fearful, surprised
    - FER2013 uses: neutral, fear, surprise
    """
    if not label:
        return None

    x = str(label).strip().lower()

    # RAVDESS -> FER2013 naming
    if x == "fearful":
        return "fear"
    if x == "surprised":
        return "surprise"
    if x == "calm":
        # FER2013 does not have "calm"; closest mapping is "neutral"
        return "neutral"

    return x


def fuse_emotions(face_emotion: str, audio_emotion: str) -> str:
    """
    Fuse face and audio emotions using a simple priority-based strategy.

    - If both modalities agree, return that emotion.
    - Otherwise, return the emotion with higher priority.
    - If either is invalid / missing, fall back to the other.
    """

    face_emotion = _normalize_emotion(face_emotion)
    audio_emotion = _normalize_emotion(audio_emotion)

    if not face_emotion and not audio_emotion:
        return "Unknown"

    if face_emotion == audio_emotion:
        return face_emotion

    # Priority tuned for typical affective importance:
    # Priorities use FER2013 label names after normalization.
    priority = {
        "angry": 5,
        "fear": 5,
        "sad": 4,
        "disgust": 4,
        "surprise": 3,
        "happy": 2,
        "neutral": 1,
    }

    if face_emotion and audio_emotion:
        if priority.get(face_emotion, 0) >= priority.get(audio_emotion, 0):
            return face_emotion
        return audio_emotion

    # If one of them is missing / invalid, use the other
    return face_emotion or audio_emotion