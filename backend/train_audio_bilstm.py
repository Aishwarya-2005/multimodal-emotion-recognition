import os
from typing import Tuple, List

import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import pickle


SR = 22050
N_MFCC = 40
MAX_PAD_LEN = 130
BATCH_SIZE = 32
EPOCHS = 60


def extract_features(file_path: str) -> np.ndarray:
    """Extract padded, silence-trimmed, and standardized MFCC features."""
    y, sr = librosa.load(file_path, sr=SR)
    
    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=20)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T

    # Cepstral Mean and Variance Normalization (CMVN)
    mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)

    if mfcc.shape[0] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:MAX_PAD_LEN, :]

    return mfcc


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Expect directory structure:
        data/audio/<emotion_label>/*.wav
    """
    features = []
    labels = []

    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if not fname.lower().endswith((".wav", ".flac", ".ogg", ".mp3")):
                continue
            path = os.path.join(label_dir, fname)
            try:
                mfcc = extract_features(path)
                features.append(mfcc)
                labels.append(label)
            except Exception as e:
                print(f"Failed to process {path}: {e}")

    X = np.array(features, dtype="float32")
    y = np.array(labels)
    return X, y, sorted(list(set(labels)))


def build_bilstm_model(
    input_shape: Tuple[int, int], num_classes: int
) -> tf.keras.Model:
    """
    Improved Hybrid Conv1D + BiLSTM model for Speech Emotion Recognition:
    - Conv1D layers extract local temporal/spectral feature representations.
    - Bidirectional LSTMs model the sequential dependency over time.
    - Dense block with BatchNorm, ELU activation, and Dropout.
    """
    reg = tf.keras.regularizers.l2(1e-4)
    inputs = layers.Input(shape=input_shape)

    # 1D Convolutional Feature Extraction Block
    x = layers.Conv1D(64, kernel_size=5, padding="same", kernel_regularizer=reg, activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", kernel_regularizer=reg, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.Dropout(0.2)(x)

    # Recurrent BiLSTM Sequential Block
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3)
    )(x)

    # Dense Classifier Block
    x = layers.Dense(128, kernel_regularizer=reg, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="audio_emotion_bilstm")

    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    data_dir = os.path.join("data", "audio")
    os.makedirs(data_dir, exist_ok=True)

    print(f"Loading audio dataset from {data_dir} ...")
    X, y_str, emotions = load_dataset(data_dir)

    if len(X) == 0:
        print("No samples found. Add audio folders to data/audio/")
        return

    print(f"Loaded {len(X)} samples, {len(emotions)} classes: {emotions}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)

    # Stratified split to preserve class distributions
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Compute class weights to handle imbalances
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Computed Class Weights:", class_weight_dict)

    model = build_bilstm_model(
        input_shape=(MAX_PAD_LEN, N_MFCC),
        num_classes=len(emotions),
    )

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=os.path.join("models", "emotion_bilstm_v1.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    os.makedirs("models", exist_ok=True)

    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[lr_scheduler, early_stop, checkpoint],
        shuffle=True,
    )

    # Save final model and label encoder
    model.save(os.path.join("models", "emotion_bilstm_last.h5"))
    with open(os.path.join("models", "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)


if __name__ == "__main__":
    main()


