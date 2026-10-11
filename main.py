import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="AgriShield | Tomato Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------
MODEL_PATH = "trained_model.h5"
MODEL_URL = (
    "https://drive.google.com/uc?"
    "id=1YNIs92QlZIJP0YQ2e5jGWUAuEWgFR_lE"
)

# Order must match the training notebook's class_names (alphabetical folders)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Indexes of the tomato classes only (this app is tomato-only)
TOMATO_IDX = [i for i, c in enumerate(CLASS_NAMES) if c.startswith("Tomato")]

TOMATO_DISEASES = {
    "Tomato___Bacterial_spot": {
        "name": "Bacterial Spot",
        "symptoms": "Small dark spots may appear on leaves and stems.",
        "advice": "Avoid overhead watering, keep foliage dry, and seek local agricultural advice."
    },
    "Tomato___Early_blight": {
        "name": "Early Blight",
        "symptoms": "Brown spots, sometimes with ring-like patterns, may appear on older leaves.",
        "advice": "Remove affected leaves carefully, avoid wetting foliage, and maintain good spacing."
    },
    "Tomato___Late_blight": {
        "name": "Late Blight",
        "symptoms": "Dark, spreading patches can develop on leaves and stems.",
        "advice": "Separate affected plants where practical and promptly consult an agricultural expert."
    },
    "Tomato___Leaf_Mold": {
        "name": "Leaf Mold",
        "symptoms": "Yellow patches may develop on upper leaf surfaces, with mold underneath.",
        "advice": "Improve air circulation, avoid excess moisture, and water near the soil."
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "symptoms": "Small circular spots with pale centers may appear on leaves.",
        "advice": "Remove badly affected leaves, reduce leaf wetness, and clean garden tools."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Two-Spotted Spider Mites",
        "symptoms": "Fine pale speckles and sometimes delicate webbing may appear on leaves.",
        "advice": "Inspect leaf undersides and seek advice about suitable pest management."
    },
    "Tomato___Target_Spot": {
        "name": "Target Spot",
        "symptoms": "Brown lesions with concentric rings may appear on leaves.",
        "advice": "Improve airflow, avoid overhead irrigation, and remove affected plant debris."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "symptoms": "Leaves may curl upward, turn yellow, and show reduced growth.",
        "advice": "Inspect for whiteflies and ask an agricultural expert about virus management."
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "symptoms": "Leaves may show mottled light and dark green areas or distorted growth.",
        "advice": "Clean hands and tools after handling plants and seek expert guidance."
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "symptoms": "The model identifies this image as belonging to the healthy tomato class.",
        "advice": "Continue regular monitoring, appropriate watering, and good garden hygiene."
    }
}

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        result = gdown.download(
            MODEL_URL,
            MODEL_PATH,
            quiet=False
        )

        if not result or not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model download failed. Check the Google Drive "
                "sharing settings and model link."
            )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs, "
            f"but {len(CLASS_NAMES)} class labels are configured."
        )

    return model


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))

    # IMPORTANT: the model was trained on raw 0-255 pixels,
    # so do NOT divide by 255 here.
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_disease(model, image):
    processed_image = prepare_image(image)
    predictions = model.predict(processed_image, verbose=0)[0]

    # Tomato-only: look only at tomato classes and renormalize
    probs = predictions[TOMATO_IDX]
    probs = probs / probs.sum()

    best = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[TOMATO_IDX[best]]
    confidence = float(probs[best])

    return predicted_class, confidence


def show_disease_info(predicted_class):
    info = TOMATO_DISEASES.get(predicted_class)

    if info is None:
        st.warning(
            "The model's top prediction is not a tomato category. "
            "Try a clear tomato leaf image."
        )
        return

    if predicted_class == "Tomato___healthy":
        st.success(f"🌿 Prediction: {info['name']}")
    else:
        st.warning(f"🔎 Prediction: {info['name']}")

    st.markdown("**Common symptoms**")
    st.write(info["symptoms"])

    st.markdown("**Suggested next steps**")
    st.write(info["advice"])

    st.caption(
        "This is an AI prediction, not a confirmed diagnosis. "
        "Check the plant and consult a local agricultural expert "
        "before taking treatment decisions."
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "page_choice" not in st.session_state:
    st.session_state["page_choice"] = "Home"


def go_to(page):
    st.session_state["page_choice"] = page


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.title("🌱 AgriShield")
    st.caption("Tomato Disease Identification")

    st.divider()

    st.radio(
        "Navigation",
        options=[
            "Home",
            "Disease Detection",
            "Tomato Care",
            "Disease Library",
            "About Project"
        ],
        key="page_choice"
    )

    st.divider()

    st.caption("AI-powered crop health support")
    st.caption("Project: AgriShield")


page = st.session_state["page_choice"]


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":
    st.title("🌱 Welcome to AgriShield")
    st.subheader("Smart Tomato Disease Identification")

    st.write(
        "AgriShield is a crop-health support website that uses "
        "a trained Convolutional Neural Network (CNN) to analyze "
        "plant leaf images and predict possible diseases."
    )

    st.info(
        "This version focuses on tomato leaf diseases and "
        "healthy tomato leaves."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Disease Detection")
        st.write(
            "Upload a tomato leaf image to see the model's prediction."
        )
        st.button(
            "Start Detection",
            on_click=go_to,
            args=("Disease Detection",),
            use_container_width=True
        )

    with col2:
        st.markdown("### 📚 Disease Library")
        st.write(
            "Learn about the tomato disease categories supported "
            "by the model."
        )
        st.button(
            "Explore Diseases",
            on_click=go_to,
            args=("Disease Library",),
            use_container_width=True
        )

    st.divider()
    st.markdown("### How AgriShield Works")

    st.markdown(
        """
        1. **Upload:** Choose a tomato leaf image.
        2. **Preprocess:** The image is resized for the model.
        3. **Predict:** The trained CNN analyzes the image.
        4. **Review:** Read the predicted category and suggested next steps.
        """
    )


# --------------------------------------------------
# DISEASE DETECTION PAGE
# --------------------------------------------------
elif page == "Disease Detection":
    st.title("🔬 Tomato Disease Detection")

    st.write(
        "Upload a clear image of a tomato leaf. For best results, "
        "use a well-lit image with the leaf clearly visible."
    )

    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            left, right = st.columns([1, 1])

            with left:
                st.image(
                    image,
                    caption="Uploaded leaf image",
                    use_container_width=True
                )

            with st.spinner("Analyzing the image..."):
                model = load_model()
                predicted_class, confidence = predict_disease(
                    model,
                    image
                )

            with right:
                st.markdown("### Analysis Result")

                st.metric(
                    "Model confidence",
                    f"{confidence * 100:.2f}%"
                )

                show_disease_info(predicted_class)

        except Exception as error:
            st.error(
                "The image could not be analyzed. "
                "Please check the model file and app logs."
            )
            st.exception(error)

    else:
        st.info(
            "Please upload a JPG, JPEG, or PNG image to begin."
        )


# --------------------------------------------------
# TOMATO CARE PAGE
# --------------------------------------------------
elif page == "Tomato Care":
    st.title("🌿 Tomato Plant Care")

    st.write(
        "Good growing practices can help reduce disease risks "
        "and support healthy tomato plants."
    )

    st.markdown("### Practical Care Tips")

    tips = [
        (
            "Water near the soil",
            "Avoid unnecessarily wetting leaves, especially late in the day."
        ),
        (
            "Allow air circulation",
            "Give plants adequate spacing and avoid overcrowding."
        ),
        (
            "Inspect leaves regularly",
            "Check both sides of leaves for unusual spots, curling, or pests."
        ),
        (
            "Keep the garden clean",
            "Remove fallen, diseased plant material appropriately."
        ),
        (
            "Clean gardening tools",
            "Clean tools between plants when disease is suspected."
        ),
        (
            "Get expert advice",
            "Confirm the problem before choosing a pesticide or other treatment."
        )
    ]

    for title, description in tips:
        with st.expander(title):
            st.write(description)

    st.warning(
        "Different diseases can have similar symptoms. "
        "Use the model result as an initial guide, not as a "
        "replacement for expert diagnosis."
    )


# --------------------------------------------------
# DISEASE LIBRARY PAGE
# --------------------------------------------------
elif page == "Disease Library":
    st.title("📚 Tomato Disease Library")

    st.write(
        "Explore the tomato categories supported by the trained model."
    )

    search_text = st.text_input(
        "Search a disease",
        placeholder="For example, Early Blight"
    )

    for class_name, info in TOMATO_DISEASES.items():
        if (
            search_text.strip().lower() in info["name"].lower()
            or search_text.strip() == ""
        ):
            with st.expander(info["name"]):
                st.write("**Common symptoms:**")
                st.write(info["symptoms"])

                st.write("**Suggested next steps:**")
                st.write(info["advice"])


# --------------------------------------------------
# ABOUT PROJECT PAGE
# --------------------------------------------------
elif page == "About Project":
    st.title("ℹ️ About AgriShield")

    st.write(
        "AgriShield is an educational project designed to demonstrate "
        "how artificial intelligence can support preliminary plant "
        "disease identification using leaf images."
    )

    st.markdown("### Technologies Used")

    st.markdown(
        """
        - **Python:** Main programming language.
        - **TensorFlow and Keras:** Load and run the trained model.
        - **CNN:** Learns visual patterns from leaf images.
        - **NumPy:** Prepares numerical image data.
        - **Pillow:** Opens and resizes uploaded images.
        - **Streamlit:** Builds the interactive web application.
        """
    )

    st.markdown("### Project Workflow")

    st.code(
        "Tomato Leaf Image\n"
        "       ↓\n"
        "Image Preprocessing\n"
        "       ↓\n"
        "Trained CNN Model\n"
        "       ↓\n"
        "Predicted Class and Confidence\n"
        "       ↓\n"
        "Disease Information and Care Suggestions",
        language="text"
    )

    st.markdown("### Important Limitation")

    st.write(
        "The model was trained on a dataset containing multiple "
        "plant categories, but this website only reports tomato "
        "results. It can still misclassify images, so predictions "
        "should be verified before acting on them."
    )

    st.caption(
        "AgriShield is an educational prototype and does not "
        "replace professional agricultural diagnosis."
    )
