
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load pretrained Siamese model (saved_model format or h5)
siamese_model = None
try:
    siamese_model = tf.keras.models.load_model("siamese_model.h5")
    print("✅ Siamese model loaded successfully.")
except Exception as e:
    print("⚠️ Model not found. Please place 'siamese_model.h5' in project root.")

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (105, 105))  # Typical size used in Siamese networks for face verification
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file1" not in request.files or "file2" not in request.files:
            return redirect(request.url)
        file1 = request.files["file1"]
        file2 = request.files["file2"]
        if file1.filename == "" or file2.filename == "":
            return redirect(request.url)

        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        path1 = os.path.join(app.config["UPLOAD_FOLDER"], filename1)
        path2 = os.path.join(app.config["UPLOAD_FOLDER"], filename2)
        file1.save(path1)
        file2.save(path2)

        similarity = None
        if siamese_model:
            img1 = preprocess_image(path1)
            img2 = preprocess_image(path2)
            pred = siamese_model.predict([img1, img2])[0][0]
            similarity = float(pred)

        return render_template("result.html", file1=filename1, file2=filename2, similarity=similarity)
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
