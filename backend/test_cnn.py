from tensorflow.keras.models import load_model

model = load_model("models/cnn_face.keras")

print("CNN Loaded Successfully")