import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

# Page Config
st.set_page_config(
    page_title="CNN CIFAR-10 Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CIFAR-10 Classes - IMPORTANT: Keep this order same as training
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

IMAGE_SIZE = 32

# Custom CSS for professional UI
def load_css():
    st.markdown("""
    <style>
   .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
   .main-header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
   .main-header p { font-size: 1.1rem; opacity: 0.9; }

   .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
   .result-card h2 { font-size: 3rem; margin: 0.5rem 0; }
   .result-card p { font-size: 1.2rem; opacity: 0.9; }

   .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }

   .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Model with caching
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model", "cnn_cifar10.keras")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    model = tf.keras.models.load_model(model_path)
    return model

# Preprocessing function - MUST match training
def preprocess_image(image):
    # Resize to 32x32
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    # Convert to array
    img_array = np.array(image)
    # Ensure 3 channels RGB
    if img_array.shape[-1] == 4: # RGBA to RGB
        img_array = img_array[..., :3]
    if len(img_array.shape) == 2: # Grayscale to RGB
        img_array = np.stack((img_array,)*3, axis=-1)

    # Normalize same as training: /255.0
    img_array = img_array.astype("float32") / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict(model, image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)[0]

    # Get top 3
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    top_3 = [(CLASSES[i], predictions[i] * 100) for i in top_3_idx]

    # Get top 1
    pred_class = CLASSES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return pred_class, confidence, top_3, predictions * 100

# Main App
def main():
    load_css()
    model = load_model()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 CNN Image Classifier</h1>
        <p>CIFAR-10 Image Classification using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    Upload an image and our trained CNN model will classify it into one of 10 CIFAR-10 categories.
    <br><b>Note:</b> This model was trained on CIFAR-10 images. Performance may vary on real-world images that differ significantly from the CIFAR-10 dataset.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About the Model")
        st.markdown("""
        **Model:** Convolutional Neural Network (CNN)
        **Dataset:** CIFAR-10
        **Task:** Image Classification
        **Classes:** 10
        **Input Size:** 32 × 32 × 3
        """)

        st.markdown("---")
        st.subheader("📦 CIFAR-10 Classes")
        for i, cls in enumerate(CLASSES):
            st.write(f"{i}. {cls.title()}")

        st.markdown("---")
        st.caption("The CNN automatically learns visual features such as edges, textures and shapes to classify images into different categories.")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.stop()

            if st.button("🔍 Classify Image", use_container_width=True, type="primary"):
                with st.spinner("Classifying..."):
                    try:
                        pred_class, confidence, top_3, all_preds = predict(model, image)
                        st.session_state['results'] = (pred_class, confidence, top_3, all_preds)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        else:
            st.info("Upload an image to start classification.")

    with col2:
        st.subheader("📊 Prediction Result")
        if 'results' in st.session_state:
            pred_class, confidence, top_3, all_preds = st.session_state['results']

            # Result Card
            st.markdown(f"""
            <div class="result-card">
                <p>PREDICTION RESULT</p>
                <h2>{pred_class.title()}</h2>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Top 3 Predictions
            st.subheader("Top 3 Predictions")
            for cls, conf in top_3:
                st.write(f"**{cls.title()}**")
                st.progress(conf / 100)
                st.caption(f"{conf:.2f}%")
        else:
            st.info("Results will appear here after classification.")

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit & TensorFlow | CIFAR-10 Dataset")

if __name__ == "__main__":
    main()
