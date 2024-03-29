import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import json

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_detection_model():
    return load_model("fmd_detection_model")

model = load_detection_model()

# Load the annotations from the JSON file
with open('fmd.json', 'r') as f:
    annotations = json.load(f)

# Title of the app
st.title("FMD Detection App")

# File uploader for image selection
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

# Check if an image is uploaded
if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Preprocess the image
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

    # Display the prediction label
    st.subheader("Prediction:")
    st.write(label)

    # Get the image size
    image_height, image_width, _ = image.shape

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
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with the bounding box
        st.subheader("FMD Detected Image:")
        st.image(image, channels="BGR")
    else:
        # Display the original image if the label is not "FMD"
        st.subheader("Original Image:")
        st.image(image, channels="BGR")
