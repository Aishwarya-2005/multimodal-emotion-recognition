import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 7  # FER2013 has 7 classes
FER2013_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


def build_face_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Deeper, regularized CNN for facial emotion recognition:
    - 4 blocks of convolutional layers (64, 128, 256, 512)
    - ELU activations for better gradient flow and zero-centered outputs
    - L2 weight decay to mitigate overfitting
    - BatchNormalization and Dropout at each layer
    """
    reg = tf.keras.regularizers.l2(1e-4)
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg, activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=reg, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=reg, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Dense Classification Block
    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_regularizer=reg, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, kernel_regularizer=reg, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="face_emotion_cnn")

    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_generators(train_dir: str, val_dir: str):
    """
    Data generators with strong but safe augmentation:
    - Rescale 1./255
    - Explicit classes parameter to match EMOTION_LABELS index order
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=FER2013_LABELS,  # Enforce explicit label-to-index mapping
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=FER2013_LABELS,  # Enforce explicit label-to-index mapping
        shuffle=False,
    )

    return train_gen, val_gen


def main():
    train_dir = os.path.join("data", "face", "train")
    val_dir = os.path.join("data", "face", "val")

    os.makedirs(os.path.dirname(train_dir), exist_ok=True)

    train_gen, val_gen = create_generators(train_dir, val_dir)

    # Sanity check: folder classes should match FER2013 labels
    found_labels = sorted(list(train_gen.class_indices.keys()))
    expected_labels = sorted(FER2013_LABELS)
    if found_labels != expected_labels:
        raise ValueError(
            "Face dataset labels mismatch.\n"
            f"Found: {found_labels}\n"
            f"Expected (FER2013): {expected_labels}\n"
            "Fix your folder names under data/face/train and data/face/val."
        )

    # Compute class weights for addressing imbalance
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Computed Class Weights:", class_weight_dict)

    model = build_face_cnn(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1),
        num_classes=len(FER2013_LABELS),
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
        filepath=os.path.join("models", "cnn_face.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    os.makedirs("models", exist_ok=True)

    model.summary()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[lr_scheduler, early_stop, checkpoint],
    )

    # Final save
    model.save(os.path.join("models", "cnn_face_last.keras"))


if __name__ == "__main__":
    main()


