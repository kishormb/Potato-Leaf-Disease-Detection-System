from flask import Flask, render_template, request, jsonify
import requests
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf  # Import TensorFlow for loading the model

app = Flask(__name__)

# Configure Upload Folder and Allowed Extensions
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Roboflow API Details
API_URL = "https://detect.roboflow.com"
API_KEY = "rfMNvVVigdGtBYwQDo8r"
MODEL_ID = "potato-disease-balnv/2"

# Dummy Model Path (Pretend this is the pre-trained model)
MODEL_PATH = 'static/models/plant_disease_model.h5'

# Ensure upload and processed directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the pre-trained model (dummy logic)."""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def predict_with_roboflow(image_path):
    """Send image to Roboflow API and return the prediction."""
    with open(image_path, "rb") as image_file:
        response = requests.post(
            f"{API_URL}/{MODEL_ID}?api_key={API_KEY}",
            files={"file": image_file}
        )
    return response.json()

def predict_with_pretrained_model(image_path, model):
    """Simulate prediction from the pre-trained model (dummy logic)."""
    # Load the image
    image = cv2.imread(image_path)
    
    # Pre-process the image (dummy pre-processing, assuming the model needs resized image)
    image_resized = cv2.resize(image, (224, 224))  # Example size (depending on your model)
    image_preprocessed = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    
    # Dummy prediction (pretending the model returns a prediction)
    # Here we simulate it by returning a fake prediction for "Healthy" or "Diseased"
    dummy_prediction = {
        "class": "Healthy" if np.random.rand() > 0.5 else "Diseased",
        "confidence": np.random.rand()
    }
    
    return dummy_prediction

def draw_polygon_trace(image_path, predictions):
    """Draw polygon traces on the image based on infected area contours."""
    image = cv2.imread(image_path)
    
    # Create a copy of the original image for drawing contours without modifying the original
    image_with_contours = image.copy()

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for infected areas (adjust based on samples)
    lower_bound = np.array([15, 50, 50])  # Lower HSV threshold
    upper_bound = np.array([35, 255, 255])  # Upper HSV threshold

    # Create mask for infected areas
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of infected regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw convex hull around infected areas (only on the image copy, not on the mask)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small detections
            hull = cv2.convexHull(contour)
            cv2.drawContours(image_with_contours, [hull], -1, (0, 0, 255), 2)  # Draw red polygon

    # Draw bounding boxes for the predictions (if any)
    for prediction in predictions:
        x, y, w, h = int(prediction["x"]), int(prediction["y"]), int(prediction["width"]), int(prediction["height"])
        
        # Draw bounding box (Red Rectangle)
        cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 0, 255), 3)

        # Add label
        label = prediction["class"]
        cv2.putText(image, label, (x - w//2, y - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save the processed image with polygon traces (convex hulls)
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(processed_path, image_with_contours)

    return processed_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(filepath)

            # Choose prediction method (Roboflow API or pre-trained model)
            use_roboflow = request.form.get('use_roboflow', 'true') == 'true'  # Default is True (Roboflow)
            
            if use_roboflow:
                # Predict using Roboflow API
                result = predict_with_roboflow(filepath)
                prediction = result.get("predictions", [{}])[0].get("class", "No disease detected")
            else:
                # Load the pre-trained model (dummy logic)
                model = load_model()

                # Predict using the pre-trained model (dummy logic)
                result = predict_with_pretrained_model(filepath, model)
                prediction = result["class"]
            
            processed_image_url = draw_polygon_trace(filepath, [{"class": prediction, "x": 100, "y": 100, "width": 50, "height": 50}])
            
            image_url = f"/static/uploads/{filename}"
            
            return jsonify({
                "image_url": image_url,
                "processed_image_url": processed_image_url,
                "prediction": prediction,
                "error": None
            })
        else:
            return jsonify({
                "error": "Invalid file type. Allowed: JPG, JPEG, PNG."
            })
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)