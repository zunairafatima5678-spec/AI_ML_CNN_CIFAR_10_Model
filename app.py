import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Configuration --- #
# Use absolute path for model loading to ensure compatibility across environments
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "cnn_cifar10.keras")
IMAGE_SIZE = (32, 32)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Custom CSS for professional UI
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .header-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .header-title {
        color: #2c3e50;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .header-subtitle {
        color: #34495e;
        font-size: 1.5em;
        text-align: center;
        margin-bottom: 20px;
    }
    .description-text {
        color: #555555;
        font-size: 1.1em;
        text-align: center;
        margin-bottom: 30px;
    }
    .upload-section {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        text-align: center;
    }
    .stFileUploader > div > div {
        background-color: #ecf0f1;
        color: #2c3e50;
        border: 2px dashed #bdc3c7;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.1em;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-size: 1.2em;
        font-weight: bold;
        margin-top: 20px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
        text-align: center;
        border-left: 8px solid #28a745;
    }
    .prediction-title {
        font-size: 1.8em;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 15px;
    }
    .predicted-class {
        font-size: 2.5em;
        font-weight: bold;
        color: #28a745;
        margin-bottom: 10px;
    }
    .confidence-score {
        font-size: 1.8em;
        color: #555555;
    }
    .top-predictions-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
        border-left: 8px solid #ffc107;
    }
    .top-predictions-title {
        font-size: 1.5em;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 15px;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db !important;
    }
    .sidebar-section {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .sidebar-title {
        font-size: 1.4em;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 15px;
    }
    .sidebar-text {
        color: #555555;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #7f8c8d;
        font-size: 0.9em;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading (with caching) --- #
@st.cache_resource
def load_cnn_model():
    """Loads the pre-trained Keras model with Streamlit caching."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}. Please ensure it is in the `model` folder.")
        st.stop()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# --- Image Preprocessing --- #
def preprocess_image(image_input):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    - Resizes to IMAGE_SIZE (32x32).
    - Converts to numpy array.
    - Normalizes pixel values to [0, 1].
    - Adds a batch dimension.
    """
    img = image_input.resize(IMAGE_SIZE, Image.LANCZOS) # Use LANCZOS for high-quality downsampling
    img_array = np.array(img).astype('float32')

    # Ensure 3 channels for RGB images
    if len(img_array.shape) == 2: # Grayscale image
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4: # RGBA image
        img_array = img_array[:, :, :3] # Discard alpha channel

    img_array /= 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# --- Prediction Function --- #
def predict_image(model, processed_img):
    """Generates predictions from the model and processes them."""
    predictions = model.predict(processed_img, verbose=0)[0] # Get probabilities for the single image
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx] * 100

    # Get top 3 predictions
    top_3_indices = np.argsort(predictions)[::-1][:3]
    top_3_predictions = [{
        "class": CLASS_NAMES[i],
        "probability": predictions[i] * 100
    } for i in top_3_indices]

    return predicted_class_idx, confidence, top_3_predictions

# --- Streamlit UI Layout --- #

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">About the Model</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="sidebar-text">
        Model: Convolutional Neural Network (CNN)<br>
        Dataset: CIFAR-10<br>
        Task: Image Classification<br>
        Classes: 10<br>
        Input Size: 32 × 32 × 3<br><br>
        The CNN automatically learns visual features such as edges,
        textures and shapes to classify images into different categories.
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">CIFAR-10 Classes</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="sidebar-text">
        <ul>
            <li>airplane</li>
            <li>automobile</li>
            <li>bird</li>
            <li>cat</li>
            <li>deer</li>
            <li>dog</li>
            <li>frog</li>
            <li>horse</li>
            <li>ship</li>
            <li>truck</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Content
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.markdown('<h1 class="header-title">CNN Image Classifier</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="header-subtitle">CIFAR-10 Image Classification using Deep Learning</h2>', unsafe_allow_html=True)
st.markdown("""
    <p class="description-text">
    This application uses a trained Convolutional Neural Network (CNN) model to classify uploaded images into one of the 10 CIFAR-10 categories.
    </p>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.write("Upload an image (JPG, JPEG, PNG) for classification.")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
            classify_button = st.button("🔍 Classify Image")

        with col2:
            if classify_button:
                # Load model
                model = load_cnn_model()
                if model is None: # Safety check if model loading failed
                    st.stop()

                # Preprocess image
                processed_img = preprocess_image(image)

                # Predict
                predicted_class_idx, confidence, top_3_predictions = predict_image(model, processed_img)
                predicted_class_name = CLASS_NAMES[predicted_class_idx]

                # Display Prediction Result Card
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown('<div class="prediction-title">PREDICTION RESULT</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="predicted-class">{predicted_class_name.upper()}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-score">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Display Top Predictions
                st.markdown('<div class="top-predictions-card">', unsafe_allow_html=True)
                st.markdown('<div class="top-predictions-title">Top Predictions</div>', unsafe_allow_html=True)
                for pred in top_3_predictions:
                    st.write(f"{pred['class'].capitalize()} ({pred['probability']:.2f}%)')
                    st.progress(pred['probability'] / 100)
                st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing image or making prediction: {e}")
        st.info("Please ensure the uploaded file is a valid image (JPG, JPEG, PNG).")
else:
    st.info("Upload an image to start classification.")

st.markdown("""
    <div style='text-align: center; margin-top: 30px; color: #7f8c8d; font-size: 0.9em;'>
        Note: This model was trained on CIFAR-10 images.
        Performance may vary on real-world images that differ significantly
        from the CIFAR-10 dataset.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2023 CIFAR-10 CNN Classifier. All rights reserved.</div>', unsafe_allow_html=True)
