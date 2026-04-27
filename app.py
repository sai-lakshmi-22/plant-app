import os
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

# ==================================================
# APP CONFIG
# ==================================================
app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY", "fallback_key")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "database.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ==================================================
# MODEL + LABELS PATH
# ==================================================
MODEL_PATH = "model/plant_model.keras"
LABELS_PATH = "model/labels.json"

# ==================================================
# LOAD LABELS
# ==================================================
labels = {}

try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    print("Labels loaded successfully")
except Exception as e:
    print("Labels load failed:", e)

# ==================================================
# SAFE LAZY MODEL LOADING
# ==================================================
model = None

def get_model():
    global model

    if model is None:
        try:
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_PATH, compile=False)
            print("Model loaded successfully:", MODEL_PATH)

        except Exception as e:
            print("Model load failed:", e)
            model = None

    return model

# ==================================================
# DATABASE MODEL
# ==================================================
class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease = db.Column(db.String(200))
    confidence = db.Column(db.String(50))

with app.app_context():
    db.create_all()

# ==================================================
# HOME PAGE
# ==================================================
@app.route("/")
def home():
    return render_template("home.html")

# ==================================================
# PREDICT PAGE
# ==================================================
@app.route("/predict", methods=["POST"])
def predict():

    mdl = get_model()

    if mdl is None:
        return jsonify({
            "status": "error",
            "message": "AI model failed to load on server"
        })

    try:
        file = request.files["image"]

        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        prediction = mdl.predict(arr)

        index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        disease_name = labels.get(str(index), f"Disease {index}")

        # Save report
        report = Report(
            disease=disease_name,
            confidence=f"{confidence:.2f}%"
        )
        db.session.add(report)
        db.session.commit()

        return jsonify({
            "status": "success",
            "disease": disease_name,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ==================================================
# REPORTS PAGE
# ==================================================
@app.route("/reports")
def reports():
    data = Report.query.order_by(Report.id.desc()).all()
    return render_template("reports.html", reports=data)

# ==================================================
# HEALTH CHECK
# ==================================================
@app.route("/health")
def health():
    return "OK"

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)