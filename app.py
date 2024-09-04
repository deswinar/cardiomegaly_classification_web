import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('Jantung_Model.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Resize to the size your model expects (adjust if your model's input size is different)
    image = image.resize((224, 224))  
    image = np.array(image)
    
    # Ensure the image has 3 channels (RGB). If the image is in grayscale (1 channel), convert it to RGB
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack((image,)*3, axis=-1)

    # Normalize the image
    image = image / 255.0  
    
    # Ensure the image has the shape (1, 224, 224, 3)
    if image.shape[-1] == 4:  # If image has an alpha channel, remove it
        image = image[..., :3]
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)  
    
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
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        result = "Cardiomegaly" if prediction[0] > 0.5 else "Normal"

        # Display the result
        st.write(f"Prediction: **{result}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
