from flask import Flask, render_template, request
from pathlib import Path
import joblib
import cv2
import numpy as np
import os
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "rice_ml_model.pkl")
label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Read Image
    image = cv2.imread(filepath)

    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    feature = image.flatten().reshape(1, -1)

    prediction = model.predict(feature)[0]
    confidence = np.max(model.predict_proba(feature)) * 100

    label = label_encoder.inverse_transform([prediction])[0]

    return render_template(
        "index.html",
        prediction=label,
        confidence=round(confidence, 2),
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)
