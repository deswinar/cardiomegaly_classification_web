import streamlit as st
import numpy as np
import joblib
import tempfile
from tensorflow.keras.models import load_model
from PIL import Image

# Define normalization ranges (these should be based on your training data)
FEATURE_RANGES = {
    'age': (0, 120),
    'education': (1, 4),
    'cigsPerDay': (0, 100),
    'totChol': (100, 600),
    'sysBP': (80, 250),
    'diaBP': (50, 150),
    'BMI': (10.0, 60.0),
    'heartRate': (30, 200),
    'glucose': (50, 400)
}

# Function to normalize features
def normalize_feature(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Function to load the .keras model from the uploaded file
def load_uploaded_keras_model(uploaded_file):
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

# Function to load the .pkl model from the uploaded file
def load_uploaded_pkl_model(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        model = joblib.load(temp_file_path)

        # Ensure that the loaded object is a valid model and has 'predict' method
        if hasattr(model, 'predict'):
            st.success("Model uploaded and loaded successfully!")
            return model
        else:
            st.error("The uploaded .pkl file is not a valid model.")
            return None
    except Exception as e:
        st.error(f"Error loading the .pkl model: {e}")
        return None

# Streamlit app
st.title("Heart Classification")

# Top navigation menu
menu = st.selectbox("Select an option", ["Cardiomegaly", "Coroner"])

if menu == "Cardiomegaly":
    st.header("Cardiomegaly Classification")
    
    uploaded_model_file = st.file_uploader("Upload your model (.h5 or .keras file)...", type=["h5", "keras"])

    if uploaded_model_file is not None:
        model = load_uploaded_keras_model(uploaded_model_file)
    else:
        model = None

    uploaded_image_file = st.file_uploader("Upload an image of the heart X-ray...", type=["jpg", "jpeg", "png"])

    if uploaded_image_file is not None and model is not None:
        try:
            image = Image.open(uploaded_image_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            prepared_image = prepare_image(image)  # Assuming you have a function to prepare the image for the model
            prediction = model.predict(prepared_image)

            st.write(f"Raw prediction values: {prediction}")

            confidence = prediction[0][0]
            result = "Normal (1)" if confidence > 0.5 else "Cardiomegaly (0)"

            st.write(f"Prediction: **{result}**")
            st.write(f"Prediction Confidence: {confidence:.2f}")

        except ValueError as e:
            st.error(f"Error processing image: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    elif uploaded_image_file is not None and model is None:
        st.warning("Please upload a model file first.")

elif menu == "Coroner":
    st.header("Coroner Prediction")

    uploaded_coroner_model_file = st.file_uploader("Upload your Coroner model (.keras or .pkl file)...", type=["keras", "pkl"])
    
    if uploaded_coroner_model_file is not None:
        if uploaded_coroner_model_file.name.endswith('.keras'):
            coroner_model = load_uploaded_keras_model(uploaded_coroner_model_file)
        elif uploaded_coroner_model_file.name.endswith('.pkl'):
            coroner_model = load_uploaded_pkl_model(uploaded_coroner_model_file)
        else:
            coroner_model = None
    else:
        coroner_model = None

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

        submitted = st.form_submit_button("Predict")
    
    if submitted:
        if coroner_model is not None:
            # Normalize input data
            normalized_age = normalize_feature(age, *FEATURE_RANGES['age'])
            normalized_education = normalize_feature(education, *FEATURE_RANGES['education'])
            normalized_cigsPerDay = normalize_feature(cigsPerDay, *FEATURE_RANGES['cigsPerDay'])
            normalized_totChol = normalize_feature(totChol, *FEATURE_RANGES['totChol'])
            normalized_sysBP = normalize_feature(sysBP, *FEATURE_RANGES['sysBP'])
            normalized_diaBP = normalize_feature(diaBP, *FEATURE_RANGES['diaBP'])
            normalized_BMI = normalize_feature(BMI, *FEATURE_RANGES['BMI'])
            normalized_heartRate = normalize_feature(heartRate, *FEATURE_RANGES['heartRate'])
            normalized_glucose = normalize_feature(glucose, *FEATURE_RANGES['glucose'])

            # Prepare input data
            input_data = np.array([[male, normalized_age, normalized_cigsPerDay, normalized_totChol, 
                                    normalized_sysBP, normalized_diaBP, normalized_BMI, normalized_heartRate, 
                                    normalized_glucose, 
                                    education == 1, education == 2, education == 3, education == 4]], dtype=np.float32)
            
            st.write(f"Normalized input data: {input_data}")

            try:
                prediction = coroner_model.predict(input_data)
                st.write(f"Raw model prediction: {prediction}")

                # Assuming binary classification with threshold 0.5
                confidence = prediction[0]

                result = "Risk of CHD (1)" if confidence > 0.5 else "No Risk of CHD (0)"
                st.write(f"Prediction: **{result}**")
                st.write(f"Prediction Confidence: {confidence:.2f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please upload a model file first.")
