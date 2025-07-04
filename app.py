import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageEnhance
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model
model = load_model("digit_model.keras")

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    probabilities = None

    if request.method == "POST":
        file = request.files["digit_image"]
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Open and preprocess
            img = Image.open(image_path).convert("L")  # Grayscale
            img = ImageEnhance.Contrast(img).enhance(2.0)  # Optional contrast boost
            img = ImageOps.fit(img, (28, 28), method=Image.Resampling.LANCZOS)
            img_array = np.array(img).astype("float32") / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            # Predict
            pred = model.predict(img_array)
            prediction = int(np.argmax(pred))
            probabilities = pred[0]

    return render_template("index.html", prediction=prediction, image_path=image_path, probs=probabilities)

if __name__ == "__main__":
    app.run(debug=True)
