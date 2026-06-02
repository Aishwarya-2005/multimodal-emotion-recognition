import librosa
import numpy as np
import pickle

from tensorflow.keras.models import load_model

model = load_model("models/emotion_bilstm_v1.h5")

with open(
    "models/label_encoder.pkl",
    "rb"
) as f:
    encoder = pickle.load(f)


def extract_features(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=22050
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc = mfcc.T

    max_pad_len = 130

    if mfcc.shape[0] < max_pad_len:

        pad_width = max_pad_len - mfcc.shape[0]

        mfcc = np.pad(
            mfcc,
            ((0, pad_width), (0, 0)),
            mode="constant"
        )

    else:
        mfcc = mfcc[:max_pad_len, :]

    return mfcc


def predict_audio_emotion(audio_path):

    features = extract_features(audio_path)

    features = np.expand_dims(
        features,
        axis=0
    )

    prediction = model.predict(features)

    emotion = encoder.inverse_transform(
        [np.argmax(prediction)]
    )[0]

    return emotion