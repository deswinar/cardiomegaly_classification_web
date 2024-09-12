import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import tempfile

# Function to load the model from the uploaded file
def load_uploaded_model(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        model = load_model(temp_file_path)
        st.success("Model uploaded and loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to preprocess the image for prediction
def prepare_image(image, target_size=(150, 150)):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if img_array.shape != (1, *target_size, 3):
        raise ValueError(f"Unexpected image shape: {img_array.shape}, expected (1, {target_size[0]}, {target_size[1]}, 3)")
    
    return img_array

# Streamlit app
st.title("Heart Classification")

# Top navigation menu
menu = st.selectbox("Select an option", ["Cardiomegaly", "Coroner"])

if menu == "Cardiomegaly":
    st.header("Cardiomegaly Classification")
    
    uploaded_model_file = st.file_uploader("Upload your model (.h5 or .keras file)...", type=["h5", "keras"])

    if uploaded_model_file is not None:
        model = load_uploaded_model(uploaded_model_file)
    else:
        model = None

    uploaded_image_file = st.file_uploader("Upload an image of the heart X-ray...", type=["jpg", "jpeg", "png"])

    if uploaded_image_file is not None and model is not None:
        try:
            image = Image.open(uploaded_image_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            prepared_image = prepare_image(image)
            prediction = model.predict(prepared_image)

            if prediction[0][0] > 0.5:
                result = "Normal (1)"
            else:
                result = "Cardiomegaly (0)"

            st.write(f"Prediction: **{result}**")
            st.write(f"Prediction Confidence: {prediction[0][0]:.2f}")

        except ValueError as e:
            st.error(f"Error processing image: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    elif uploaded_image_file is not None and model is None:
        st.warning("Please upload a model file first.")

elif menu == "Coroner":
    st.header("Coroner Prediction")
    
    # Upload model for Coroner prediction
    uploaded_coroner_model_file = st.file_uploader("Upload your Coroner model (.keras file)...", type=["keras"])
    
    if uploaded_coroner_model_file is not None:
        coroner_model = load_uploaded_model(uploaded_coroner_model_file)
    else:
        coroner_model = None

    # Input fields for the Coroner features
    with st.form("coroner_form"):
        st.write("Input the following features:")
        male = st.selectbox("Male (0 = No, 1 = Yes):", [0, 1])
        age = st.number_input("Age:", min_value=0, max_value=120, value=50)
        education = st.selectbox("Education (1-4):", [1, 2, 3, 4])
        currentSmoker = st.selectbox("Current Smoker (0 = No, 1 = Yes):", [0, 1])
        cigsPerDay = st.number_input("Cigarettes per Day:", min_value=0, max_value=100, value=0)
        BPMeds = st.selectbox("Blood Pressure Meds (0 = No, 1 = Yes):", [0, 1])
        prevalentStroke = st.selectbox("Prevalent Stroke (0 = No, 1 = Yes):", [0, 1])
        prevalentHyp = st.selectbox("Prevalent Hypertension (0 = No, 1 = Yes):", [0, 1])
        diabetes = st.selectbox("Diabetes (0 = No, 1 = Yes):", [0, 1])
        totChol = st.number_input("Total Cholesterol (mg/dL):", min_value=100, max_value=600, value=200)
        sysBP = st.number_input("Systolic Blood Pressure (mm Hg):", min_value=80, max_value=250, value=120)
        diaBP = st.number_input("Diastolic Blood Pressure (mm Hg):", min_value=50, max_value=150, value=80)
        BMI = st.number_input("Body Mass Index (kg/m^2):", min_value=10.0, max_value=60.0, value=25.0)
        heartRate = st.number_input("Heart Rate (bpm):", min_value=30, max_value=200, value=70)
        glucose = st.number_input("Glucose (mg/dL):", min_value=50, max_value=400, value=100)
        TenYearCHD = st.selectbox("Ten Year CHD (0 = No, 1 = Yes):", [0, 1])

        # Submit button
        submitted = st.form_submit_button("Predict")
    
    # Prediction logic
    if submitted:
        if coroner_model is not None:
            # Create a numpy array from input data
            input_data = np.array([[male, age, education, currentSmoker, cigsPerDay, BPMeds,
                                    prevalentStroke, prevalentHyp, diabetes, totChol, sysBP,
                                    diaBP, BMI, heartRate, glucose, TenYearCHD]], dtype=np.float32)
            
            # Check input data shape
            st.write(f"Input data shape: {input_data.shape}")
            
            # Check model input shape
            try:
                model_input_shape = coroner_model.input_shape
                st.write(f"Model expected input shape: {model_input_shape}")
            except Exception as e:
                st.error(f"Unable to determine model input shape: {e}")

            # Make a prediction
            try:
                prediction = coroner_model.predict(input_data)
                # Display prediction results
                result = "Risk of CHD (1)" if prediction[0][0] > 0.5 else "No Risk of CHD (0)"
                st.write(f"Prediction: **{result}**")
                st.write(f"Prediction Confidence: {prediction[0][0]:.2f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please upload a model file first.")
