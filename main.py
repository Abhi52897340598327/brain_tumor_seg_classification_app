import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_config
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import json

# Paths to the uploaded files
config_path = "/classification/config.json"
weights_path = "/classification/model.weights.h5"

# Load model configuration and weights
def load_model_from_config_and_weights(config_path, weights_path):
    # Load the config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Recreate the model
    model = tf.keras.models.model_from_config(config['config'])
    
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
    classification_model = load_model_from_config_and_weights(config_path, weights_path)
    st.success("Classification model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the classification model: {e}")
    st.stop()

# Class labels
class_labels = ["Glioma", "Meningioma", "Pituitary Tumor"]

# Streamlit app
st.title("Brain Tumor Classification")
st.markdown("Upload an MRI image to classify the type of brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display the image
    st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
    
    # Process the image
    try:
        image = Image.open(uploaded_file).convert("RGB")
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
    except Exception as e:
        st.error(f"Error processing the image or making predictions: {e}")
