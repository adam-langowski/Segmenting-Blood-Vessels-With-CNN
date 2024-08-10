import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from io import BytesIO
import base64
from PIL import Image

# Check for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")
else:
    print("No GPU found, using CPU")

# Load models
models = {
    "HRF": tf.keras.models.load_model('model_segmentingVessels.keras'),
    "DRIVE": tf.keras.models.load_model('model_DRIVE_segmentingVessels.keras')
}

def segment_image(image, model):
    target_height = 584
    target_width = 876
    resized_image = cv2.resize(image, (target_width, target_height))
    resized_image = np.expand_dims(resized_image, axis=0) / 255.0
    prediction = model.predict(resized_image)
    return prediction[0]

def apply_threshold(pred, threshold=0.42):
    binary_image = np.where(pred >= threshold, 255, 0).astype(np.uint8)
    return binary_image

def get_image_download_link(img_array, filename, text):
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# UI config
st.set_page_config(
    page_title="RBVS",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown("<h1 style='text-align: center; color: white;'>Retinal Blood Vessel Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Upload an image and select a model to segment retinal blood vessels</p>", unsafe_allow_html=True)

# model selection
st.sidebar.markdown("<h2 style='color: white;'>Model Selection</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: gray; font-size: 16px;'>There are 2 models available, trained on 2 different databases.</p>", unsafe_allow_html=True)

model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ("HRF", "DRIVE")
)

# file uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# segmenting
if uploaded_file is not None:
    st.markdown("<h5 style='color: #50586C;'>Original Image:</h3>", unsafe_allow_html=True)
    st.image(uploaded_file, use_column_width=True)
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    model = models.get(model_choice, models['HRF'])

    with st.spinner("Segmenting..."):
        prediction = segment_image(image, model)
        segmented_image = apply_threshold(prediction)

    st.markdown("<h5 style='color: #50586C;'>Segmented Image:</h3>", unsafe_allow_html=True)
    st.image(segmented_image, use_column_width=True)
    #st.markdown(get_image_download_link(segmented_image, 'segmented_image.png', 'Download Segmented Image'), unsafe_allow_html=True)

# Footer
st.markdown("""
    <style>
    footer {
        visibility: hidden;
    }
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #50586C;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        font-family: Arial, sans-serif;
    }
    </style>
    <div class="footer">
        <p>© 2024 Retinal Blood Vessels Segmentation | Created by Adam Langowski</p>
    </div>
    """, unsafe_allow_html=True)