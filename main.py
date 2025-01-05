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

    # Define custom objects if needed
    custom_objects = {
        'SpatialAttention': SpatialAttention,
        'NormalizeLayer': NormalizeLayer
    }

    for idx, layer_config in enumerate(config['config']['layers']):
        layer_class_name = layer_config['class_name']
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
    elif page == "Segmentation":
        st.title("Segmentation Page")

if __name__ == "__main__":
    main()
