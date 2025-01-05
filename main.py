import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import *
import numpy as np
from custom_layers import SpatialAttention, NormalizeLayer
from PIL import Image, UnidentifiedImageError
import json
import zipfile
import os
import shutil

# Paths to models and decompression directories
classification_config_path = "classification/config.json"
classification_weights_path = "classification/model.weights.h5"
segmentation_zip_path = "segmentation_model_compressed.zip"
segmentation_extract_path = "segmentation"
segmentation_config_path = os.path.join(segmentation_extract_path, 'segmentation_model_take_5_config.json')
segmentation_weights_path = os.path.join(segmentation_extract_path, 'segmentation_model_take_5_weights.weights.h5')

# Decompression Function
def decompress_segmentation_model(zip_path, extract_path):
    if not os.path.exists(extract_path):  # Only decompress if not already done
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        st.success("Segmentation model decompressed successfully!")

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
    return model

# Load Segmentation Model
def rebuild_segmentation_model(config_path, weights_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    input_shape = config['config']['layers'][0]['config']['batch_shape'][1:]  # Exclude batch size
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    layer_outputs = {}

    # Define custom objects if needed
    custom_objects = {
        'SpatialAttention': SpatialAttention,  # Replace with your custom implementation
        'NormalizeLayer': NormalizeLayer  # Replace with your custom implementation
    }

    for idx, layer_config in enumerate(config['config']['layers']):
        layer_class_name = layer_config['class_name']
        # Check if layer class exists in tf.keras.layers or in custom_objects
        layer_class = getattr(tf.keras.layers, layer_class_name, custom_objects.get(layer_class_name))
        if not layer_class:
            raise ValueError(f"Layer class {layer_class_name} not found. Ensure it is part of tf.keras.layers or custom_objects.")
        layer = layer_class.from_config(layer_config['config'])
        if isinstance(layer, tf.keras.layers.InputLayer):
            layer_outputs[idx] = x
            continue
        elif isinstance(layer, tf.keras.layers.Concatenate):
            inbound_nodes = layer_config.get('inbound_nodes', [])
            inputs_to_concat = [layer_outputs[node_idx[0]] for node_idx in inbound_nodes[0]]
            x = layer(inputs_to_concat)
        else:
            x = layer(x)
        layer_outputs[idx] = x
    outputs = x
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.load_weights(weights_path)
    return model

# Prediction Functions
def classify_image(image, classification_model):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = classification_model.predict(image_array)
    return predictions

def segment_image(image, segmentation_model):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = segmentation_model.predict(image_array)
    prediction = np.squeeze(prediction)
    return prediction

# App Pages
def classification_page(classification_model):
    st.title("Brain Tumor Classification")
    uploaded_file = st.file_uploader("Upload an MRI image for classification", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded MRI Image", use_column_width=True)
            predictions = classify_image(image, classification_model)
            predicted_class = ["Glioma", "Meningioma", "Pituitary Tumor"][np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            st.subheader("Prediction")
            st.write(f"**Predicted Tumor Type:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
        except UnidentifiedImageError:
            st.error("Invalid image file. Please upload a valid PNG, JPG, or JPEG file.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

def segmentation_page(segmentation_model):
    st.title("Brain Tumor Segmentation")
    uploaded_file = st.file_uploader("Upload an MRI image for segmentation", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded MRI Image", use_column_width=True)
            segmented_image = segment_image(image, segmentation_model)
            st.subheader("Segmentation Result")
            st.image(segmented_image, caption="Segmented Tumor Mask", use_column_width=True)
        except UnidentifiedImageError:
            st.error("Invalid image file. Please upload a valid PNG, JPG, or JPEG file.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# Main App Function
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Classification", "Segmentation"])
    
    # Decompress the segmentation model
    decompress_segmentation_model(segmentation_zip_path, segmentation_extract_path)
    
    # Load models
    classification_model = rebuild_classification_model(classification_config_path, classification_weights_path)
    segmentation_model = rebuild_segmentation_model(segmentation_config_path, segmentation_weights_path)
    
    if page == "Classification":
        classification_page(classification_model)
    elif page == "Segmentation":
        segmentation_page(segmentation_model)

if __name__ == "__main__":
    main()
