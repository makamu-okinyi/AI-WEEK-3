# app.py

import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Create a route for the main page
@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

# Create a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    """Receives drawing data, processes it, and returns a prediction."""
    # Get the image data from the request
    data = request.get_json()
    image_data = data['image_data']

    # Decode the base64 image
    # The data URL is in the format "data:image/png;base64,iVBORw0KGgo..."
    # We need to strip the header part to get the pure base64 data
    img_str = image_data.split(',')[1]
    img_bytes = base64.b64decode(img_str)
    
    # Open the image using Pillow
    img = Image.open(io.BytesIO(img_bytes))

    # --- Preprocess the image for the model ---
    # 1. Convert to grayscale
    img = img.convert('L')
    # 2. Resize to 28x28 pixels
    img = img.resize((28, 28))
    # 3. Convert to a NumPy array
    img_array = np.array(img)
    # 4. Reshape for the model (add batch and channel dimensions)
    processed_img = img_array.reshape(1, 28, 28, 1)
    # 5. Normalize the pixel values
    processed_img = processed_img.astype('float32') / 255.0

    # --- Make a prediction ---
    prediction = model.predict(processed_img)
    predicted_digit = int(np.argmax(prediction)) # Convert to a standard Python int
    confidence = float(np.max(prediction))     # Convert to a standard Python float

    # Return the prediction as JSON
    return jsonify({
        'predicted_digit': predicted_digit,
        'confidence': confidence
    })

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)