import streamlit as st
import numpy as np
import joblib
import tempfile
from tensorflow.keras.models import load_model
from PIL import Image

# Define normalization ranges (based on your training data)
FEATURE_RANGES = {
    'age': (0, 120),
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

# Function to load the .pkl model
def load_uploaded_pkl_model(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        model = joblib.load(temp_file_path)

        if hasattr(model, 'predict_proba'):
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

menu = st.selectbox("Select an option", ["Coroner"])

if menu == "Coroner":
    st.header("Coroner Prediction")

    uploaded_coroner_model_file = st.file_uploader("Upload your Coroner model (.pkl file)...", type=["pkl"])
    
    if uploaded_coroner_model_file is not None:
        coroner_model = load_uploaded_pkl_model(uploaded_coroner_model_file)
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
            normalized_cigsPerDay = normalize_feature(cigsPerDay, *FEATURE_RANGES['cigsPerDay'])
            normalized_totChol = normalize_feature(totChol, *FEATURE_RANGES['totChol'])
            normalized_sysBP = normalize_feature(sysBP, *FEATURE_RANGES['sysBP'])
            normalized_diaBP = normalize_feature(diaBP, *FEATURE_RANGES['diaBP'])
            normalized_BMI = normalize_feature(BMI, *FEATURE_RANGES['BMI'])
            normalized_heartRate = normalize_feature(heartRate, *FEATURE_RANGES['heartRate'])
            normalized_glucose = normalize_feature(glucose, *FEATURE_RANGES['glucose'])

            # One-hot encode 'education' feature
            education_1 = 1 if education == 1 else 0
            education_2 = 1 if education == 2 else 0
            education_3 = 1 if education == 3 else 0
            education_4 = 1 if education == 4 else 0

            # Prepare input data (18 features total)
            input_data = np.array([[male, normalized_age, currentSmoker, normalized_cigsPerDay, 
                                    BPMeds, prevalentStroke, prevalentHyp, diabetes, normalized_totChol, 
                                    normalized_sysBP, normalized_diaBP, normalized_BMI, normalized_heartRate, 
                                    normalized_glucose, education_1, education_2, education_3, education_4]], 
                                    dtype=np.float32)
            
            st.write(f"Normalized input data: {input_data}")

            try:
                prediction_proba = coroner_model.predict_proba(input_data)
                st.write(f"Raw model prediction probabilities: {prediction_proba}")

                # Assuming binary classification with threshold 0.5
                confidence = prediction_proba[0][1]  # Probability for class 1

                result = "Risk of CHD (1)" if confidence > 0.5 else "No Risk of CHD (0)"
                st.write(f"Prediction: **{result}**")
                st.write(f"Prediction Confidence: {confidence:.2f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please upload a model file first.")
