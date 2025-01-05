import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import numpy as np
from PIL import Image, UnidentifiedImageError
import json
import zipfile
import os

# Paths to models and decompression directories
classification_config_path = "classification/config.json"
classification_weights_path = "classification/model.weights.h5"
segmentation_zip_path = "segmentation_model_compressed.zip"
segmentation_extract_path = "segmentation"
segmentation_config_path = os.path.join(segmentation_extract_path, 'segmentation_model_take_5_config.json')
segmentation_weights_path = os.path.join(segmentation_extract_path, 'segmentation_model_take_5_weights.weights.h5')

# Define placeholder custom layers (replace with actual implementations)
class SpatialAttention(tf.keras.layers.Layer):
    def call(self, inputs):
        # Dummy implementation
        return inputs

class NormalizeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Dummy implementation
        return inputs

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

    # Define custom objects
    custom_objects = {
        'SpatialAttention': SpatialAttention,
        'NormalizeLayer': NormalizeLayer
    }

    for idx, layer_config in enumerate(config['config']['layers']):
        layer_class_name = layer_config['class_name']
        layer_class = getattr(tf.keras.layers, layer_class_name, custom_objects.get(layer_class_name))
        if not layer_class:
            raise ValueError(f"Layer class {layer_class_name} not found in tf.keras.layers or custom_objects.")
        layer = layer_class.from_config(layer_config['config'])

        if isinstance(layer, tf.keras.layers.InputLayer):
            layer_outputs[idx] = x
            continue

        elif isinstance(layer, tf.keras.layers.Concatenate):
            inbound_nodes = layer_config.get('inbound_nodes', [])
            if not inbound_nodes or not isinstance(inbound_nodes, list):
                raise ValueError(f"Concatenate layer requires valid 'inbound_nodes'. Found: {inbound_nodes}")

            valid_inputs = []
            for node in inbound_nodes:
                args = node.get('args', [])
                if isinstance(args, list):
                    for input_item in args[0]:  # Each input item in args
                        if isinstance(input_item, dict) and 'keras_history' in input_item.get('config', {}):
                            keras_history = input_item['config']['keras_history']
                            ref_layer_index = keras_history[1]  # Reference the layer index
                            if ref_layer_index in layer_outputs:
                                valid_inputs.append(layer_outputs[ref_layer_index])

            if not valid_inputs:
                raise KeyError(
                    f"No valid inputs for 'Concatenate' layer at index {idx}. "
                    f"Inbound nodes: {inbound_nodes}. Available layer_outputs keys: {list(layer_outputs.keys())}"
                )
            x = layer(valid_inputs)

        else:
            x = layer(x)

        # Save the output of the current layer
        layer_outputs[idx] = x

    outputs = x
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.load_weights(weights_path)
    return model

# Main App
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Classification", "Segmentation"])

    # Decompress the segmentation model
    decompress_segmentation_model(segmentation_zip_path, segmentation_extract_path)

    # Load models
    classification_model = rebuild_classification_model(classification_config_path, classification_weights_path)
    segmentation_model = rebuild_segmentation_model(segmentation_config_path, segmentation_weights_path)

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
