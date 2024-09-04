import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the saved model
model = load_model('Jantung_Model.h5')  # Ensure the path is correct or place the model in the same directory as app.py

# Function to preprocess the image for prediction
def prepare_image(image, target_size=(150, 150)):
    """
    Preprocesses the uploaded image to be compatible with the model's input.

    Args:
    - image: PIL Image object
    - target_size: tuple, the target size of the image as expected by the model

    Returns:
    - Preprocessed image array suitable for model prediction
    """
    # Resize and convert the image to an array
    img = image.resize(target_size)
    img_array = img_to_array(img)
    # Normalize the image array
    img_array = img_array / 255.0
    # Expand dimensions to match the input shape (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app
st.title("Heart Classification (Normal or Cardiomegaly)")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload an image of the heart X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    prepared_image = prepare_image(image)

    # Make prediction
    prediction = model.predict(prepared_image)

    # Output prediction (0 = Normal, 1 = Cardiomegaly)
    if prediction[0][0] > 0.5:
        result = "Cardiomegaly (1)"
    else:
        result = "Normal (0)"

    # Display the result
    st.write(f"Prediction: **{result}**")
    st.write(f"Prediction Confidence: {prediction[0][0]:.2f}")
