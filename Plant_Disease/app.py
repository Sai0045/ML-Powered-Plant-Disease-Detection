import os
import numpy as np
import requests
from flask import Flask, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Blynk Configuration
BLYNK_TOKEN = "wK3Eh5_fLVfzP8unYHYJa2QZb1eU2tIe"  # Replace with your actual token
BLYNK_DISEASE_VPIN = "V1"  # Virtual pin name (set in Blynk dashboard)

# Load YOLOv8 model
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found!")

print(f"✅ Loading YOLOv8 model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Global variables
latest_result = "No Disease Detected"
latest_details = "The plant appears to be healthy."
latest_filename = None

# Utility functions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        return True
    except Exception as e:
        print(f"❌ Invalid image file: {e}")
        return False

def is_blank_image(filepath, threshold=30):
    try:
        img = Image.open(filepath).convert("L")
        mean_val = np.mean(np.array(img))
        print(f"🖼 Image brightness mean: {mean_val}")
        return mean_val < threshold
    except Exception as e:
        print(f"❌ Error checking blank image: {e}")
        return False

def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")

# Routes
@app.route("/", methods=["GET"])
def home():
    return "<h2>🌿 Smart Plant Disease Detection Server is Running!</h2>", 200

@app.route("/upload", methods=["POST"])
def upload_image():
    global latest_result, latest_details, latest_filename

    latest_result = "Processing..."
    latest_details = "Processing..."
    latest_filename = None

    if "file" not in request.files:
        latest_result = "No file received"
        latest_details = "No file part found in the request."
        return jsonify({
            "status": "error",
            "result": latest_result,
            "details": latest_details
        }), 400

    file = request.files["file"]
    if file.filename == "":
        latest_result = "No file selected"
        latest_details = "No file was selected for upload."
        return jsonify({
            "status": "error",
            "result": latest_result,
            "details": latest_details
        }), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamped_filename = f"{get_timestamp()}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], timestamped_filename)
        file.save(filepath)

        if not is_valid_image(filepath):
            latest_result = "Invalid image"
            latest_details = "Uploaded file is not a valid image."
            return jsonify({
                "status": "error",
                "result": latest_result,
                "details": latest_details
            }), 400

        if is_blank_image(filepath):
            latest_result = "Blank or very dark image detected"
            latest_details = "The uploaded image appears to be blank or too dark to analyze."
            return jsonify({
                "status": "warning",
                "result": latest_result,
                "details": latest_details
            }), 200

        latest_filename = timestamped_filename

        try:
            results = model(filepath)
            detections = results[0].boxes
            details = []

            for box in detections:
                conf = float(box.conf.item())
                if conf > 0.6:
                    cls_id = int(box.cls.item())
                    label = model.names[cls_id]
                    print(f"Detected: {label} with confidence {conf}")
                    details.append(label.replace(" ", ""))  # Shortened label

            if details:
                latest_result = "Disease Detected"
                latest_details = ",".join(details)
                if len(latest_details) > 100:
                    latest_details = latest_details[:100] + "..."
            else:
                latest_result = "No Disease Detected"
                latest_details = "Healthy"

            # ✅ Send to Blynk
            try:
                blynk_url = f"https://blynk-cloud.com/{BLYNK_TOKEN}/update/{BLYNK_DISEASE_VPIN}?value={latest_details}"
                response = requests.get(blynk_url)
                if response.status_code == 200:
                    print(f"📤 Sent result to Blynk: {latest_details}")
                else:
                    print(f"⚠ Blynk responded with status code: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"❌ Failed to send to Blynk: {e}")

            return jsonify({
                "status": "success",
                "result": latest_result,
                "details": latest_details
            }), 200

        except Exception as e:
            latest_result = "Error during inference"
            latest_details = str(e)
            print(f"❌ Inference Error: {e}")
            return jsonify({
                "status": "error",
                "result": latest_result,
                "details": latest_details
            }), 500

    latest_result = "Invalid file type"
    latest_details = "Only JPG, JPEG, and PNG files are allowed."
    return jsonify({
        "status": "error",
        "result": latest_result,
        "details": latest_details
    }), 400

@app.route("/result", methods=["GET"])
def get_result():
    return jsonify({
        "result": latest_result,
        "details": latest_details
    }), 200

@app.route("/image", methods=["GET"])
def get_image():
    if latest_filename:
        return send_from_directory(app.config["UPLOAD_FOLDER"], latest_filename)
    return jsonify({
        "status": "error",
        "result": "No image found"
    }), 404

@app.route("/health", methods=["GET"])
def health_check():
    return "OK", 200

# Start server
if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    print("🚀 Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5001)
