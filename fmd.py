import os
import zipfile
import requests
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st

# Function to load the detection model
@st.cache(allow_output_mutation=True)
def load_detection_model(model_dir="./"):
    # Define the URL of the model zip file on GitHub
    model_url = "https://github.com/GEOFFREY-MO/DSAIL-ATTACHMENT/raw/main/fmd_detection_model.zip"
    
    # Download the model zip file
    r = requests.get(model_url)
    with open("fmd_detection_model.zip", "wb") as f:
        f.write(r.content)
    
    # Extract the model zip file
    with zipfile.ZipFile("fmd_detection_model.zip", 'r') as zip_ref:
        zip_ref.extractall(model_dir)
    
    # Load the model
    model_path = os.path.join(model_dir, "fmd_detection_model")
    return load_model(model_path)

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

# Upload image or take picture
st.sidebar.title("Upload Image or Take Picture")
option = st.sidebar.selectbox(
    'Choose an option:',
    ('Upload Image', 'Take Picture')
)

if option == 'Upload Image':
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
else:
    camera = cv2.VideoCapture(0)
    if st.sidebar.button('Take Picture'):
        _, image = camera.read()
        st.image(image, caption='Captured Image', use_column_width=True)
    camera.release()

# Process the image and make predictions
if st.sidebar.button('Detect'):
    if option == 'Upload Image' or option == 'Take Picture':
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)

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

        # Print the prediction
        st.write("Prediction:", label)

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
