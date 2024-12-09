import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image, UnidentifiedImageError
import json

# Paths to the uploaded files
config_path = "classification/config.json"
weights_path = "classification/model.weights.h5"

# Load model configuration and weights
def rebuild_model_from_config_and_weights(config_path, weights_path):
    # Load the config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Rebuild the model manually using the layers from config
    model = Sequential()
    for layer_config in config['config']['layers']:
        layer_class = getattr(tf.keras.layers, layer_config['class_name'])
        layer = layer_class.from_config(layer_config['config'])
        model.add(layer)

    # Load weights
    model.load_weights(weights_path)

    # Compile the model if compile_config exists
    if 'compile_config' in config:
        compile_config = config['compile_config']
        optimizer = tf.keras.optimizers.get(compile_config['optimizer'])
        loss = compile_config['loss']
        metrics = compile_config['metrics']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

# Load the model
try:
    classification_model = rebuild_model_from_config_and_weights(config_path, weights_path)
    # st.success("Classification model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the classification model: {e}")
    st.stop()

# Class labels
class_labels = ["Glioma", "Meningioma", "Pituitary Tumor"]

# Streamlit app
st.title("Brain Tumor Classification")
st.markdown("""
### Welcome to the Brain Tumor Classification System
This application is designed to assist in identifying the type of brain tumor based on MRI scans. Please provide the required details below and upload the MRI image to proceed.

**All rights reserved by Team AiMinds.**
""")

# Collect patient details
st.sidebar.header("Patient Information")
with st.sidebar.form("patient_form"):
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    address = st.text_area("Address")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    phone_number = st.text_input("Phone Number")
    email = st.text_input("Email Address")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    st.sidebar.write(f"**Name:** {name}")
    st.sidebar.write(f"**Age:** {age}")
    st.sidebar.write(f"**Gender:** {gender}")
    st.sidebar.write(f"**Phone:** {phone_number}")
    st.sidebar.write(f"**Email:** {email}")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # Open the uploaded file as an image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        # Preprocess the image for the model
        image = image.resize((128, 128))  # Resize to match model input
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = classification_model.predict(image_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display results
        st.subheader("Prediction")
        st.write(f"Predicted Tumor Type: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a PNG, JPG, or JPEG file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
