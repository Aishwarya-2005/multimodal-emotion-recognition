import pickle

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

print(encoder.classes_)