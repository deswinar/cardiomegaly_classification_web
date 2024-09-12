import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import pandas as pd
import io
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Function to load the model from the uploaded file
def load_uploaded_model(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_file:
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
    st.header("Coroner - Heart Disease Analysis Using Framingham Dataset")
    
    # Upload dataset
    uploaded_file = st.file_uploader("Upload Framingham Heart Study dataset (CSV format)...", type=["csv"])

    if uploaded_file is not None:
        # Load the dataset
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(data.head())

            # Basic Statistics
            st.write("Dataset Statistics:")
            st.write(data.describe())

            # Visualize Data
            st.write("Visualize Data:")
            fig, ax = plt.subplots()
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            # Data Preparation for Model
            st.write("Model Training:")
            data = data.dropna()  # Handle missing values for simplicity

            # Feature Selection
            X = data.drop('TenYearCHD', axis=1)
            y = data['TenYearCHD']

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Logistic Regression Model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Display Results
            st.write("Model Accuracy:", accuracy_score(y_test, predictions))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, predictions))
            st.write("Classification Report:")
            st.text(classification_report(y_test, predictions))

        except Exception as e:
            st.error(f"Error loading or processing the dataset: {e}")
    else:
        st.info("Please upload the Framingham Heart Study dataset in CSV format.")
