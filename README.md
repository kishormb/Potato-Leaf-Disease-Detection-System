# 🥔 Potato Leaf Disease Detection using CNN + Flask

This project is a web-based application that detects **potato leaf diseases** (Early Blight, Late Blight, and Healthy) using a **Convolutional Neural Network (CNN)** model built with **TensorFlow**. The interface is powered by **Flask**, allowing users to upload images and receive instant predictions.

---

## 🚀 Features

- Upload an image of a potato leaf through a simple web interface.
- CNN model classifies the image into:
  - **Healthy**
  - **Early Blight**
  - **Late Blight**
- Real-time prediction and result display.
- Clean and responsive front-end using Flask templates.

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS (Jinja2 templating via Flask)
- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow / Keras (CNN)
- **Image Processing**: OpenCV
- **Other**: NumPy, Werkzeug, Requests

---

## 🧠 How It Works

1. User uploads a potato leaf image via the web interface.
2. The image is preprocessed using OpenCV (resizing, normalization, etc.).
3. The trained CNN model predicts the class of the disease.
4. The result is displayed back to the user with a confidence score.

---

## 📁 Project Structure

├── static/ # CSS, JS, Images ├── templates/ # HTML templates │ └── index.html ├── uploads/ # Uploaded leaf images ├── model/ # Trained CNN model (e.g., model.h5) ├── app.py # Main Flask app ├── README.md # This file └── requirements.txt # Python dependencies
---

## 📷 Sample Output

*Add screenshots of the UI and sample prediction results here.*

---

## 🔧 Setup Instructions

1. **Clone the Repository**

## bash
git clone https://github.com/yourusername/potato-leaf-disease-flask.git 
cd potato-leaf-disease-flask 

Install Dependencies
pip install -r requirements.txt
Add Your Trained Model
Place your trained Keras model (e.g., model.h5) in the /model directory.

Run the Flask App
python app.py
Then, open your browser and go to http://127.0.0.1:5000
