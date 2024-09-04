import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

# Function to load the model from the uploaded file
def load_uploaded_model(uploaded_file):
    """
    Loads a Keras model from an uploaded file.

    Args:
    - uploaded_file: Uploaded file object from Streamlit

    Returns:
    - Loaded Keras model
    """
    try:
        # Read the file as bytes and load the model using BytesIO
        model = load_model(io.BytesIO(uploaded_file.read()))
        st.success("Model uploaded and loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

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
    # Ensure the image is in RGB mode (3 channels)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize and convert the image to an array
    img = image.resize(target_size)
    img_array = img_to_array(img)

    # Normalize the image array
    img_array = img_array / 255.0
    
    # Expand dimensions to match the input shape (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Check shape
    if img_array.shape != (1, *target_size, 3):
        raise ValueError(f"Unexpected image shape: {img_array.shape}, expected (1, {target_size[0]}, {target_size[1]}, 3)")
    
    return img_array

# Streamlit app
st.title("Heart Classification (Normal or Cardiomegaly)")

# File uploader for the user to upload their model
uploaded_model_file = st.file_uploader("Upload your model (.h5 or .keras file)...", type=["h5", "keras"])

if uploaded_model_file is not None:
    # Load the uploaded model
    model = load_uploaded_model(uploaded_model_file)
else:
    model = None

# File uploader for the user to upload an image
uploaded_image_file = st.file_uploader("Upload an image of the heart X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_image_file is not None and model is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_image_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        prepared_image = prepare_image(image)

        # Make prediction
        prediction = model.predict(prepared_image)

        # Output prediction (0 = Cardiomegaly, 1 = Normal)
        if prediction[0][0] > 0.5:
            result = "Normal (1)"
        else:
            result = "Cardiomegaly (0)"

        # Display the result
        st.write(f"Prediction: **{result}**")
        st.write(f"Prediction Confidence: {prediction[0][0]:.2f}")

    except ValueError as e:
        st.error(f"Error processing image: {e}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

elif uploaded_image_file is not None and model is None:
    st.warning("Please upload a model file first.")
