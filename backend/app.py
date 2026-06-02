from face_service import predict_face_emotion
from audio_service import predict_audio_emotion
from fusion_service import fuse_emotions

from flask import Flask, render_template, request, jsonify

import os
import uuid
import traceback

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict from uploaded image and audio.
    Saves uploaded files and executes emotion prediction using deep learning models.
    Always returns JSON response for dynamic UI updates.
    """
    image = request.files.get("image")
    audio = request.files.get("audio")

    if not image or not audio:
        return jsonify({
            "success": False,
            "error": "Both a face image and an audio file are required."
        }), 400

    # Use random filenames to avoid collisions
    image_filename = f"{uuid.uuid4().hex}_{image.filename or 'image.png'}"
    audio_filename = f"{uuid.uuid4().hex}_{audio.filename or 'audio.wav'}"

    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)

    try:
        image.save(image_path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Failed to save uploaded face image: {str(e)}"
        }), 500

    try:
        audio.save(audio_path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Failed to save uploaded audio file: {str(e)}"
        }), 500

    # Perform facial emotion inference
    try:
        face_emotion, face_confidence = predict_face_emotion(image_path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Face emotion model execution failed: {str(e)}"
        }), 500

    # Perform audio emotion inference
    try:
        audio_emotion, audio_confidence = predict_audio_emotion(audio_path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Audio emotion model execution failed: {str(e)}"
        }), 500

    # Fuse predictions (pass raw emotion strings)
    try:
        final_emotion = fuse_emotions(face_emotion, audio_emotion)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Fusion decision service failed: {str(e)}"
        }), 500

    return jsonify({
        "success": True,
        "face": face_emotion,
        "face_confidence": face_confidence,
        "audio": audio_emotion,
        "audio_confidence": audio_confidence,
        "result": final_emotion
    })


if __name__ == "__main__":
    app.run(debug=True)