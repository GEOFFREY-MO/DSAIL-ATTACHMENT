import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import streamlit as st
from PIL import Image
import os
import zipfile

# Function to load the model
def load_detection_model():
    # Check if the model directory exists
    model_dir = "model"
    if not os.path.exists(model_dir):
        # Download the model file from GitHub
        os.system("wget https://github.com/GEOFFREY-MO/DSAIL-ATTACHMENT/raw/main/fmd_detection_model.zip")
        # Extract the model file
        with zipfile.ZipFile("fmd_detection_model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir)
    # Load the model
    return load_model(os.path.join(model_dir, "fmd_detection_model"))

# Load the trained model
model = load_detection_model()

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# Load the annotations from the JSON file
with open('fmd.json', 'r') as f:
    annotations = json.load(f)

# Streamlit app
st.title("FMD Detection App")

# Upload image or capture from camera
uploaded_file = st.file_uploader("Upload Image or Capture from Camera", type=["jpg", "jpeg", "png"])

# Process uploaded image or camera capture
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)

    # Get the predicted class (0 or 1)
    predicted_class = np.argmax(predictions)

    # Set a threshold for unknown class detection
    threshold = 0.4

    # Check if the maximum predicted probability is below the threshold
    if np.max(predictions) < threshold:
        label = "Unknown"
    else:
        # Map the predicted class to the corresponding label
        if predicted_class == 0:
            label = "FMD"
        else:
            label = "No FMD"

    # Display prediction label
    st.write(f"Prediction: {label}")

    # Get the image size
    image_height, image_width, _ = processed_image.shape

    # Draw the bounding box only if the label is "FMD"
    if label == "FMD":
        # Find the annotation that closely matches the image size
        for annotation in annotations['annotations']:
            bbox = annotation['bbox']
            if bbox:
                x, y, w, h = bbox
                if image_width * 0.5 < w < image_width * 1.5 and image_height * 0.5 < h < image_height * 1.5:
                    break
        else:
            # If no matching annotation found, draw a dummy bounding box covering 3/4 of the image
            x = int(image_width * 0.125)  # 1/8 of the width
            y = int(image_height * 0.125)  # 1/8 of the height
            w = int(image_width * 0.75)  # 3/4 of the width
            h = int(image_height * 0.75)  # 3/4 of the height

        # Draw the bounding box
        cv2.rectangle(processed_image[0], (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label the bounding box
        cv2.putText(processed_image[0], "FMD Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image with the bounding box
        st.image(processed_image[0], caption='FMD Detected', use_column_width=True)
    else:
        # Display the original image if the label is not "FMD"
        st.image(processed_image[0], caption='Original Image', use_column_width=True)
