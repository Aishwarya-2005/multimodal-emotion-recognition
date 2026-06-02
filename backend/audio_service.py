import librosa
import pickle
import numpy as np

from pathlib import Path

from tensorflow.keras.models import load_model

_BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = _BASE_DIR / "models" / "emotion_bilstm_v1.h5"
ENCODER_PATH = _BASE_DIR / "models" / "label_encoder.pkl"
MAX_PAD_LEN = 130
N_MFCC = 40
SR = 22050

model = load_model(str(MODEL_PATH))

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)


def extract_features(audio_path: str) -> np.ndarray:
    """Extract padded, trimmed, and standardized MFCC features."""
    signal, sr = librosa.load(audio_path, sr=SR)

    # Trim leading/trailing silence to focus on speech segments
    signal, _ = librosa.effects.trim(signal, top_db=20)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T

    # Cepstral Mean and Variance Normalization (CMVN)
    mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)

    if mfcc.shape[0] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:MAX_PAD_LEN, :]

    return mfcc


def predict_audio_emotion(audio_path: str):
    """Predict emotion from audio file path, returning (emotion, confidence)."""
    try:
        features = extract_features(audio_path)
    except Exception as e:
        print(f"Error during audio feature extraction: {e}")
        return "Unknown", 0.0

    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features, verbose=0)
    
    emotion_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    if hasattr(encoder, "inverse_transform"):
        try:
            emotion = encoder.inverse_transform([emotion_idx])[0]
            return emotion, round(confidence, 2)
        except Exception as e:
            print(f"Inverse transform failed: {e}")

    # Fallback if encoder is not standard
    classes = getattr(encoder, "classes_", None)
    if classes is not None and 0 <= emotion_idx < len(classes):
        return str(classes[emotion_idx]), round(confidence, 2)

    return "Unknown", round(confidence, 2)