import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('Jantung_Model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the size your model expects
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("Heart Classification (Normal or Cardiomegaly)")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    result = "Cardiomegaly" if prediction[0] > 0.5 else "Normal"

    # Display the result
    st.write(f"Prediction: **{result}**")
