import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image, ImageOps, ImageEnhance
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = None  # global model (lazy-loaded)

@app.route("/", methods=["GET", "POST"])
def index():
    global model
    prediction = None
    image_url = None
    probabilities = None

    if request.method == "POST":
        file = request.files.get("digit_image")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Lazy load tensorflow and the model
            if model is None:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                from tensorflow.keras.models import load_model
                model = load_model("digit_model.h5", compile=False)

            # Preprocess image
            img = Image.open(filepath).convert("L")
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = ImageOps.fit(img, (28, 28), method=Image.Resampling.LANCZOS)
            img_array = np.array(img).astype("float32") / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            # Prediction
            pred = model.predict(img_array)
            prediction = int(np.argmax(pred))
            probabilities = pred[0]
            image_url = f"/uploads/{filename}"

    return render_template("index.html", prediction=prediction, image_url=image_url, probs=probabilities)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
