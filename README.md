
# 🖊️ Handwritten Digit Recognition Web App

This is a Flask-based web application that allows users to upload an image of a handwritten digit (0–9), and it predicts the digit using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

---

## 🚀 Features

- Upload handwritten digit images (JPG, PNG, etc.)
- Processes the image using PIL and NumPy
- Predicts the digit using a `.keras` model trained on MNIST
- Displays top-10 class probabilities
- Lightweight Flask backend with HTML frontend
- Saves uploaded images in a dedicated `/uploads` folder (outside `/static`)

---

## 📁 Project Structure

```
Handwritten-Digit-Recognition/
├── app.py                  # Main Flask application
├── digit_model.keras       # Pretrained CNN model (MNIST)
├── uploads/                # Uploaded images (created dynamically)
├── templates/
│   └── index.html          # HTML frontend
├── static/
│   └── style.css           # Optional custom CSS
├── train_model.py          # Script to train model (optional)
├── .gitignore
└── README.md
```

---

## 🧠 Tech Stack

- **Python 3.8+**
- **Flask** – Web framework
- **TensorFlow / Keras** – Deep learning
- **Pillow (PIL)** – Image preprocessing
- **NumPy** – Array operations

---

## 🛠️ Setup Instructions

### 1. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

(Or manually install: `flask`, `tensorflow`, `pillow`, `numpy`)

### 2. 🧠 Add or train your model

Ensure `digit_model.keras` is in the root folder. You can use your own trained model or run `train_model.py` (if available).

### 3. ▶️ Run the Flask app

```bash
python app.py
```

### 4. 🌐 Open in browser

Visit: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🧪 Example

1. Upload a clear image of a handwritten digit.
2. App processes and resizes it to 28x28 grayscale.
3. The CNN predicts the digit and shows confidence scores.

---

## 🔐 .gitignore (Recommended)

```gitignore
uploads/
__pycache__/
*.pyc
*.keras
*.h5
.env
```

---

## 📸 Screenshot (Optional)

You can add a screenshot here:
```markdown
![Screenshot](static/screenshot.png)
```

---

## 🙋 Author

Developed by [Mayank Aggarwal (Mayank-01x)](https://github.com/Mayank-01x)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
