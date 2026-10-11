
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown

# -------------------- Configuration --------------------

st.set_page_config(
    page_title="AgriShield | Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

MODEL_PATH = "trained_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1YNIs92QlZIJP0YQ2e5jGWUAuEWgFR_lE"

# -------------------- Load Trained Model --------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading the trained model. Please wait..."):
            downloaded_file = gdown.download(
                MODEL_URL,
                MODEL_PATH,
                quiet=False
            )

        if not downloaded_file or not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model download failed. Please check Google Drive sharing permissions."
            )

        if os.path.getsize(MODEL_PATH) < 1_000_000:
            os.remove(MODEL_PATH)
            raise RuntimeError(
                "The downloaded file is too small to be the model. "
                "Please check that the Google Drive link points to the model file."
            )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# -------------------- Disease Prediction --------------------

def model_prediction(test_image):
    model = load_model()

    image = tf.keras.preprocessing.image.load_img(
        test_image,
        target_size=(128, 128)
    )

    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)
    result_index = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0])) * 100

    return result_index, confidence


# -------------------- Disease Classes --------------------

class_name = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# -------------------- Sidebar --------------------

st.sidebar.title("🌿 AgriShield")
st.sidebar.write("Plant Disease Detection")
app_mode = st.sidebar.selectbox(
    "Select Page",
    ["Home", "About", "Disease Recognition"]
)


# -------------------- Home Page --------------------

if app_mode == "Home":
    st.title("🌿 AgriShield")
    st.subheader("Plant Disease Detection and Preventive Measures")

    image_path = "home_page.jpeg"

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

    st.markdown("""
    Welcome to **AgriShield**, a plant disease recognition system
    designed to help identify possible diseases from plant leaf images.

    ### How It Works
    1. **Upload:** Choose an image of a plant leaf.
    2. **Analyze:** Our trained deep-learning model analyzes the image.
    3. **Identify:** View the predicted plant disease.

    ### Why AgriShield?
    - Quick image-based disease prediction
    - Simple and user-friendly interface
    - Supports 38 plant health and disease categories

    **Note:** Predictions are model estimates, not a substitute for
    expert agricultural advice. Confirm important decisions with a
    qualified agricultural specialist.
    """)


# -------------------- About Page --------------------

elif app_mode == "About":
    st.title("About AgriShield")

    st.markdown("""
    AgriShield is a machine-learning project that uses a
    Convolutional Neural Network (CNN) to classify plant leaf images.

    ### Dataset Information
    - Training images: 70,295
    - Validation images: 17,572
    - Categories: 38
    - Input image size: 128 × 128 pixels

    ### Technology Used
    - Python
    - TensorFlow and Keras
    - Convolutional Neural Network (CNN)
    - NumPy
    - Streamlit

    ### Project Goal
    To help users recognize possible plant diseases from leaf images
    and encourage timely consultation and appropriate crop care.
    """)


# -------------------- Disease Recognition Page --------------------

elif app_mode == "Disease Recognition":
    st.title("🔍 Plant Disease Recognition")

    st.write("Upload a clear image of a plant leaf to predict its category.")

    test_image = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if test_image is not None:
        st.image(
            test_image,
            caption="Uploaded plant image",
            use_container_width=True
        )

        if st.button("Predict Disease", type="primary"):
            try:
                with st.spinner("Analyzing the image..."):
                    result_index, confidence = model_prediction(test_image)

                st.success(
                    f"Prediction: {class_name[result_index].replace('___', ' - ').replace('_', ' ')}"
                )
                st.write(f"Model confidence: {confidence:.2f}%")

                st.info(
                    "This is an AI-generated prediction. Image quality, "
                    "lighting, and unfamiliar field conditions can affect "
                    "results. Please verify the diagnosis before taking action."
                )

            except Exception as error:
                st.error(
                    "Prediction could not be completed. Please check the "
                    "model download and deployment logs."
                )
                st.caption(f"Technical details: {error}")
