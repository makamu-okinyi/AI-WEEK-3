# app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

# Set page configuration
st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")

# Function to load the model (cached to prevent reloading on every interaction)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    return model

model = load_model()

# --- UI Layout ---
st.title("✏️ Handwritten Digit Recognizer")
st.markdown("Draw a digit from 0 to 9 in the canvas below, and the CNN model will try to guess what it is!")

# Create a two-column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Drawing Canvas")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=20, # Brush size
        stroke_color="#FFFFFF", # White brush
        background_color="#000000", # Black background
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    # Display a placeholder until a drawing is made
    prediction_text = st.empty()
    prediction_text.info("The model's prediction will appear here.")
    
    # Add a "Predict" button
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            # 1. Preprocess the image
            # Get image from canvas, convert to grayscale
            img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            # Resize to 28x28
            resized_img = cv2.resize(img, (28, 28))
            # Reshape for the model (add batch and channel dimensions)
            processed_img = resized_img.reshape(1, 28, 28, 1)
            # Normalize the image
            processed_img = processed_img / 255.0

            # 2. Make a prediction
            prediction = model.predict(processed_img)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)

            # 3. Display the result
            prediction_text.success(f"Predicted Digit: **{predicted_digit}** with {confidence:.2%} confidence.")

            # Optional: Display the processed image
            st.write("Processed Image (28x28) fed to the model:")
            st.image(resized_img, caption="Model Input")
        else:
            prediction_text.warning("Please draw a digit first!")