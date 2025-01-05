import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError
import os

# Paths to models
classification_config_path = "classification/config.json"
classification_weights_path = "classification/model.weights.h5"
segmentation_model_path = "small_segmentation_model.keras"  # Use the trained model file

# Load Classification Model
def rebuild_classification_model(config_path, weights_path):
    import json
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import *

    with open(config_path, 'r') as f:
        config = json.load(f)
    model = Sequential()
    for layer_config in config['config']['layers']:
        layer_class = getattr(tf.keras.layers, layer_config['class_name'])
        layer = layer_class.from_config(layer_config['config'])
        model.add(layer)
    model.load_weights(weights_path)
    return model

# Main App
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Classification", "Segmentation"])

    # Load models
    classification_model = rebuild_classification_model(classification_config_path, classification_weights_path)
    segmentation_model = load_model(segmentation_model_path)  # Load your trained segmentation model

    if page == "Classification":
        st.title("Classification Page")
        st.write("Classification model loaded successfully!")
        uploaded_file = st.file_uploader("Upload an image for classification", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                # Preprocess the image
                image_array = np.array(image.resize((128, 128))) / 255.0
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                # Predict
                prediction = classification_model.predict(image_array)
                st.write(f"Classification Prediction: {np.argmax(prediction)}")
            except UnidentifiedImageError:
                st.error("Error: Invalid image file!")

    elif page == "Segmentation":
        st.title("Segmentation Page")
        st.write("Segmentation model loaded successfully!")
        uploaded_file = st.file_uploader("Upload an image for segmentation", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                # Preprocess the image
                image_array = np.array(image.resize((128, 128))) / 255.0
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                # Predict
                predicted_mask = segmentation_model.predict(image_array)
                st.image(predicted_mask.squeeze(), caption="Predicted Mask", use_column_width=True, clamp=True, channels="GRAY")
            except UnidentifiedImageError:
                st.error("Error: Invalid image file!")

if __name__ == "__main__":
    main()
