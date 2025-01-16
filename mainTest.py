import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('BrainTumor10EpochsCategorical', compile=False)

# Specify the image path
image_path = r'C:\Users\A2Z\Desktop\brain tumour detection\dataset\pred\pred0.jpg'

# Read the image using OpenCV
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    print(f"Error: Could not load image at {image_path}")
else:
    # Resize the image to match the input size of the model
    img = Image.fromarray(image)
    img = img.resize((64, 64))

    # Normalize pixel values
    img = (np.array(img) / 255.0).astype(np.float32)

    # Expand dimensions to match the model's expected input shape
    input_img = np.expand_dims(img, axis=0)

    # Make predictions using the loaded model
    predictions = model.predict(input_img)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)

    # Display the result
    print(f"Predicted Class Index: {predicted_class_index}")
