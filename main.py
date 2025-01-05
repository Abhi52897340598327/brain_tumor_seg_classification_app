import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image, UnidentifiedImageError
import json

# Paths to models
classification_config_path = "classification/config.json"
classification_weights_path = "classification/model.weights.h5"
segmentation_model_path = "small_segmentation_model.keras"  # Trained segmentation model

# Load Classification Model
def rebuild_classification_model(config_path, weights_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = Sequential()
    for layer_config in config['config']['layers']:
        layer_class = getattr(tf.keras.layers, layer_config['class_name'])
        layer = layer_class.from_config(layer_config['config'])
        model.add(layer)

    model.load_weights(weights_path)

    # Compile if needed
    if 'compile_config' in config:
        compile_config = config['compile_config']
        optimizer = tf.keras.optimizers.get(compile_config['optimizer'])
        loss = compile_config['loss']
        metrics = compile_config['metrics']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

# Class labels for classification
class_labels = ["Glioma", "Meningioma", "Pituitary Tumor"]

# Load models
try:
    classification_model = rebuild_classification_model(classification_config_path, classification_weights_path)
except Exception as e:
    st.error(f"Error loading the classification model: {e}")
    st.stop()

try:
    segmentation_model = load_model(segmentation_model_path)
except Exception as e:
    st.error(f"Error loading the segmentation model: {e}")
    st.stop()

# App Title and Description
st.title("NeuroLens: Brain Tumor Classification and Segmentation")
st.markdown("""
### Welcome to NeuroLens
This application classifies the type of brain tumor and segments its location from MRI scans.

**Please fill in the patient details in the sidebar and upload an MRI image for analysis.**
""")

# Patient Information
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

# File uploader for MRI image
uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

        # Preprocess the image for both models
        preprocessed_image = image.resize((128, 128))  # Resize to model input size
        image_array = np.array(preprocessed_image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Step 1: Classification
        st.subheader("Classification Results")
        classification_predictions = classification_model.predict(image_array)
        predicted_class = class_labels[np.argmax(classification_predictions)]
        classification_confidence = np.max(classification_predictions) * 100
        st.write(f"**Predicted Tumor Type:** {predicted_class}")
        st.write(f"**Confidence:** {classification_confidence:.2f}%")

        # Step 2: Segmentation
        st.subheader("Segmentation Results")
        segmentation_predictions = segmentation_model.predict(image_array)
        st.image(segmentation_predictions.squeeze(), caption="Predicted Tumor Segmentation Mask", use_container_width=True, clamp=True, channels="GRAY")

        # Step 3: Combined Summary
        st.subheader("Summary")
        st.write(f"Patient **{name}**, aged **{age}**, has been diagnosed with a predicted brain tumor type of **{predicted_class}** with a confidence of **{classification_confidence:.2f}%**.")
        st.write("The segmentation mask for the tumor has been successfully generated above.")

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a PNG, JPG, or JPEG file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
