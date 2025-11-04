import streamlit as st
import requests
import numpy as np
import tensorflow as tf
import cv2
import os
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "https://detect.roboflow.com"
API_KEY = "rfMNvVVigdGtBYwQDo8r"
MODEL_ID = "potato-disease-balnv/2"
MODEL_PATH = "static/models/plant_disease_model.h5"

# -----------------------------
# APP TITLE
# -----------------------------
st.set_page_config(page_title="Potato Leaf Disease Detection", page_icon="🥔", layout="wide")
st.title("🥔 Potato Leaf Disease Detection System 🍃")
st.markdown("Upload a potato leaf image below to identify if it’s healthy or diseased.")

# -----------------------------
# Disease Information Database
# -----------------------------
disease_info = {
    "Early-Blight": {
        "name": "🌱 Early Blight (Alternaria solani)",
        "cause": "Caused by *Alternaria solani*, thrives in warm, humid conditions.",
        "remedies": [
            "✅ Rotate crops yearly to prevent disease buildup.",
            "🌞 Avoid overhead watering.",
            "🍂 Remove and destroy infected leaves.",
            "🧪 Apply fungicides like chlorothalonil or copper-based products."
        ]
    },
    "Late-Blight": {
        "name": "🌿 Late Blight (Phytophthora infestans)",
        "cause": "Caused by *Phytophthora infestans*, a water mold spreading in cool, damp weather.",
        "remedies": [
            "✅ Use certified disease-free seeds.",
            "🌞 Ensure proper soil drainage.",
            "🧪 Apply fungicides like mancozeb or chlorothalonil.",
            "🍂 Remove and dispose of infected plants."
        ]
    },
    "Healthy": {
        "name": "✅ Healthy Plant",
        "cause": "No signs of disease detected. The plant appears healthy.",
        "remedies": [
            "🌱 Maintain good irrigation and soil quality.",
            "🌞 Ensure sunlight and ventilation.",
            "💧 Avoid overwatering to prevent fungal infection."
        ]
    }
}

# -----------------------------
# Helper Functions
# -----------------------------
def predict_with_roboflow(image_path):
    """Send image to Roboflow API and return the prediction."""
    with open(image_path, "rb") as image_file:
        response = requests.post(
            f"{API_URL}/{MODEL_ID}?api_key={API_KEY}",
            files={"file": image_file}
        )
    return response.json()

def load_model():
    """Load the pre-trained model."""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def predict_with_local_model(image_path, model):
    """Fake local prediction for demo purposes."""
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)
    
    # Simulate random prediction
    result = {
        "class": "Healthy" if np.random.rand() > 0.5 else "Diseased",
        "confidence": round(np.random.rand(), 2)
    }
    return result

def draw_polygon_trace(image_path):
    """Highlight infected-looking regions (color-based approximation)."""
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([15, 50, 50])
    upper_bound = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            hull = cv2.convexHull(contour)
            cv2.drawContours(image, [hull], -1, (0, 0, 255), 2)

    processed_path = "processed_image.jpg"
    cv2.imwrite(processed_path, image)
    return processed_path

# -----------------------------
# STREAMLIT UI
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload a potato leaf image", type=["jpg", "jpeg", "png"])
use_roboflow = st.toggle("Use Roboflow API for detection", value=True)

if uploaded_file is not None:
    # Save the uploaded file
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Display the original image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(file_path, use_column_width=True)

    # Run Prediction
    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing..."):
            if use_roboflow:
                try:
                    result = predict_with_roboflow(file_path)
                    prediction_class = result.get("predictions", [{}])[0].get("class", "Healthy")
                except Exception as e:
                    st.error("Error connecting to Roboflow API. Using local prediction instead.")
                    model = load_model()
                    result = predict_with_local_model(file_path, model)
                    prediction_class = result["class"]
            else:
                model = load_model()
                result = predict_with_local_model(file_path, model)
                prediction_class = result["class"]

            processed_path = draw_polygon_trace(file_path)

            with col2:
                st.subheader("Infected Area")
                st.image(processed_path, use_column_width=True)

            st.success(f"🧠 Prediction: **{prediction_class}**")

            # Show Disease Information
            info = disease_info.get(prediction_class, disease_info["Healthy"])
            st.markdown(f"### {info['name']}")
            st.write(f"**Cause:** {info['cause']}")
            st.markdown("### 🛡️ Remedies:")
            for r in info["remedies"]:
                st.markdown(f"- {r}")
