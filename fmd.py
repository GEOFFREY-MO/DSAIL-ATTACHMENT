import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
from google.colab.patches import cv2_imshow

# Load the trained model
model = load_model("/content/fmd_detection_model")

# Define a function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# Load the annotations from the JSON file
with open('/content/fmd.json', 'r') as f:
    annotations = json.load(f)

# Define the image path
image_path = "/content/fmd19.JPG"

# Preprocess the image
processed_image = preprocess_image(image_path)

# Make predictions
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
print("Prediction:", label)

# Get the image size
image = cv2.imread(image_path)
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
    cv2_imshow(image)
else:
    # Display the original image if the label is not "fmd"
    cv2_imshow(image)
