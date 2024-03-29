import os
import zipfile
import requests
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# Function to load the detection model
def load_detection_model(model_dir="./"):
    # Define the URL of the model zip file on GitHub
    model_url = "https://github.com/GEOFFREY-MO/DSAIL-ATTACHMENT/blob/3c7766fdf9a08cef1ee47cd46b038d506a23a172/fmd_detection_model.zip"
    
    # Download the model zip file
    r = requests.get(model_url)
    with open("fmd_detection_model.zip", "wb") as f:
        f.write(r.content)
    
    # Extract the model zip file
    with zipfile.ZipFile("fmd_detection_model.zip", 'r') as zip_ref:
        zip_ref.extractall(model_dir)
    
    # Load the model
    model_path = os.path.join(model_dir, "fmd_detection_model")
    return tf.saved_model.load(model_path)

# Load the trained model
model = load_detection_model()

# Define a function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# Load the annotations from the JSON file
def load_annotations(annotations_url):
    r = requests.get(annotations_url)
    annotations = r.json()
    return annotations

# Define the URL of the annotations JSON file on GitHub
annotations_url = "https://raw.githubusercontent.com/GEOFFREY-MO/DSAIL-ATTACHMENT/main/fmd.json"
annotations = load_annotations(annotations_url)

# Function to make predictions and display result
def predict_and_display(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model(processed_image)

    # Get the predicted class (0 or 1)
    predicted_class = np.argmax(predictions)

    # Set a threshold for unknown class detection
    threshold = 0.4

    # Check if the maximum predicted probability is below the threshold
    if np.max(predictions) < threshold:
        label = "unknown"
    else:
        # Map the predicted class to the corresponding label
        if predicted_class == 0:
            label = "fmd"
        else:
            label = "no_fmd"

    # Get the image size
    image_height, image_width, _ = image.shape

    # Draw the bounding box only if the label is "fmd"
    if label == "fmd":
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
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label the bounding box
        cv2.putText(image, "FMD detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image with the bounding box
        st.image(image, caption='FMD detected', use_column_width=True)
    else:
        # Display the original image if the label is not "fmd"
        st.image(image, caption='Original Image', use_column_width=True)

# Title for the app
st.title("FMD Detection App")

# Add a file uploader for uploading images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Add an image picker for taking a picture
if st.button("Take a picture"):
    st.write("Not implemented yet")  # Placeholder for taking a picture using webcam

# If an image is uploaded, process and display it
if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    predict_and_display(image)
