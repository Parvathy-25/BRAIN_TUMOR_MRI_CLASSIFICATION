import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🧠 Brain Tumor MRI Classifier")
st.sidebar.write("Upload an MRI scan image to detect tumor type.")
st.sidebar.write("Model: Custom CNN")
st.sidebar.write("Input Size: 128 x 128")

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("brain_tumor_cnn_model.h5")

model = load_cnn_model()

# Confirm model input shape
st.sidebar.write("Model Input Shape:", model.input_shape)

# Class Labels (MUST match training folder order)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --------------------------------------------------
# Main Title
# --------------------------------------------------
st.title("Brain Tumor MRI Classification System")

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # -------------------------
        # Open Image & Convert RGB
        # -------------------------
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        # -------------------------
        # Preprocessing
        # -------------------------
        img = np.array(image)

        # Resize to match training
        img = cv2.resize(img, (128, 128))

        # Normalize (same as training)
        img = img.astype("float32") / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        st.write("Image shape sent to model:", img.shape)

        # -------------------------
        # Prediction
        # -------------------------
        prediction = model.predict(img)

        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        # -------------------------
        # Display Results
        # -------------------------
        st.subheader("Prediction Result")

        st.success(
            f"Predicted Tumor Type: {class_names[predicted_class]}"
        )

        st.info(f"Confidence: {confidence:.2f}%")

        # -------------------------
        # Probability Chart
        # -------------------------
        st.subheader("Prediction Probabilities")

        probabilities = prediction[0] * 100

        fig = plt.figure()
        plt.bar(class_names, probabilities)
        plt.xlabel("Tumor Type")
        plt.ylabel("Probability (%)")
        plt.title("Class Probability Distribution")
        plt.xticks(rotation=45)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing image: {e}")
