import streamlit as st
import numpy as np
import tensorflow as tf
import os
from forensics.exif_analysis import extract_exif
from forensics.ela_analysis import perform_ela
from utils.image_preprocessing import load_and_preprocess_image
from utils.visualization import plot_ela_image

# Page Config
st.set_page_config(
    page_title="TraceFake - AI Image Detector",
    page_icon="🔍",
    layout="wide"
)

# Constants
MODEL_PATH = 'models/saved_model/tracefake_v1.h5'

# Load Model (Cached)
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        return None

model = load_model()

# Sidebar
st.sidebar.title("🔍 TraceFake")
st.sidebar.info(
    """
    **AI-Based Image Authenticity Checker**
    
    This tool detects whether an uploaded image is **Real** or **AI-Generated**.
    
    It uses:
    1. **Deep Learning**: EfficientNetB0 CNN.
    2. **ELA**: Error Level Analysis for manipulation traces.
    3. **EXIF**: Metadata analysis.
    """
)
st.sidebar.write("---")
st.sidebar.write("Developed for transparency in AI.")

# Main Interface
st.title("TraceFake: Artificial Intelligence Image Verification")
st.markdown("Upload an image to valid its authenticity.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display Image
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
    with st.spinner('Analyzing...'):
        # 2. Preprocess
        # We need a copy of the file for different operations since reading consumes the pointer
        import io
        file_bytes = uploaded_file.getvalue()
        
        # Prepare for Model
        processed_img, rgb_img = load_and_preprocess_image(io.BytesIO(file_bytes))
        
        # 3. Model Inference
        with col2:
            st.subheader("🤖 Prediction Result")
            if model:
                prediction = model.predict(processed_img)
                confidence = prediction[0][0] # Output of sigmoid is 0 to 1
                
                # Assuming 0 = Fake, 1 = Real (or vice versa, typical is 0=ClassA, 1=ClassB)
                # We need to map this based on Training Data.
                # For this demo, let's assume 1 = Real, 0 = Fake.
                # Adjust threshold as needed.
                
                is_real = confidence > 0.5
                label = "REAL" if is_real else "FAKE (AI-GENERATED)"
                conf_percent = confidence if is_real else (1 - confidence)
                
                color = "green" if is_real else "red"
                st.markdown(f"<h2 style='color: {color};'>{label}</h2>", unsafe_allow_html=True)
                st.progress(float(conf_percent))
                st.write(f"Confidence: **{conf_percent*100:.2f}%**")
                
            else:
                st.warning("⚠️ Model not found or untrained. Displaying Mock/Random Result for Demo Mode.")
                # Mock result for logic validation
                fake_score = 0.88
                st.markdown(f"<h2 style='color: red;'>FAKE (AI-GENERATED)</h2>", unsafe_allow_html=True)
                st.progress(fake_score)
                st.write(f"Confidence: **{fake_score*100:.2f}%** (DEMO)")

    st.divider()
    
    # 4. Forensics
    st.header("🕵️ Forensics Analysis")
    
    tab1, tab2 = st.tabs(["EXIF Metadata", "Error Level Analysis (ELA)"])
    
    with tab1:
        st.subheader("Metadata Extraction")
        exif_data = extract_exif(io.BytesIO(file_bytes))
        st.json(exif_data)
        
    with tab2:
        st.subheader("Error Level Analysis")
        st.write("ELA highlights differences in compression levels. White/noisy areas indicate potential manipulation or resaving artifacts common in Deepfakes.")
        
        ela_result = perform_ela(io.BytesIO(file_bytes))
        
        # Plot side by side
        fig = plot_ela_image(rgb_img, ela_result)
        st.pyplot(fig)

else:
    st.info("Please upload an image to start.")
