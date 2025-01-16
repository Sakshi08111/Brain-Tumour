from flask import Flask, render_template, request 
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model('BrainTumor10EpochsCategorical', compile=False)

def process_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        return None
    
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img = (np.array(img) / 255.0).astype(np.float32)
    input_img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(input_img)
    predicted_class_index = np.argmax(predictions)
    
    return predicted_class_index

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('index.html', result=result, error="No file part")
        
        file = request.files['file']
        
        # Check if the file is selected
        if file.filename == '':
            return render_template('index.html', result=result, error="No selected file")
        
        try:
            # Save the file to a temporary location
            file_path = 'temp_image.jpg'
            file.save(file_path)
            
            # Process the image using the model
            result = process_image(file_path)
        except Exception as e:
            return render_template('index.html', result=result, error=f"Error processing image: {str(e)}")
    
    return render_template('index.html', result=result, error=None)

if __name__ == '__main__':
    app.run(debug=True)


    
