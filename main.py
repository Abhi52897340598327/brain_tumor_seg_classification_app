import streamlit as st
import numpy as np
from PIL import Image
import cv2
from yaml import safe_load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import CustomObjectScope

keras.config.enable_unsafe_deserialization()

# Load the models
classification_model = load_model('/Users/abhiraamvenigalla/PycharmProjects/brain_tumor_seg_classification_app/brain_tumor_classification_model.keras', safe_mode = False)
segmentation_model = load_model('/Users/abhiraamvenigalla/PycharmProjects/brain_tumor_seg_classification_app/segmentation_unet_model.keras')  # Adjust path if needed

# Define class labels
class_labels = ["Glioma", "Meningioma", "Pituitary Tumor"]

# App title and description
st.title("NeuroLens")
st.markdown("""
This application uses AI models to classify brain tumors into categories or segment tumor areas in MRI images.
""")

# Collect patient details in a sidebar form
st.sidebar.subheader("Patient Information")
with st.sidebar.form("patient_form"):
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    dob = st.date_input("Date of Birth")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    ethnicity = st.text_input("Ethnicity")
    medical_history = st.text_area("Past Medical History (if any)")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    st.sidebar.write(f"Patient Name: {name}")
    st.sidebar.write(f"Age: {age}")
    st.sidebar.write(f"Date of Birth: {dob}")
    st.sidebar.write(f"Gender: {gender}")
    st.sidebar.write(f"Ethnicity: {ethnicity}")
    st.sidebar.write(f"Medical History: {medical_history}")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)

    # Load image for processing
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((128, 128))  # Ensure size matches model input
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Batch dimension

    # Model options
    task = st.radio("Select Task", ["Tumor Classification", "Tumor Segmentation"])

    if task == "Tumor Classification":
        # Predict tumor type
        predictions = classification_model.predict(image_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display classification result
        st.subheader("Classification Result")
        st.write(f"Predicted Tumor Type: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

    elif task == "Tumor Segmentation":
        # Perform segmentation
        segmented_output = segmentation_model.predict(image_array)
        segmented_image = (segmented_output[0, :, :, 0] * 255).astype(np.uint8)  # Adjust mask

        # Display segmentation results
        st.subheader("Segmentation Result")
        st.image(segmented_image, caption="Tumor Segmentation", use_column_width=True)
