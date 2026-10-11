
import os
from io import BytesIO

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =========================================================
# AGRISHIELD | TOMATO HEALTH ASSISTANT
# =========================================================

st.set_page_config(
    page_title="AgriShield | Tomato Health",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "trained_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1YNIs92QlZIJP0YQ2e5jGWUAuEWgFR_lE"

# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --forest: #244B35;
    --leaf: #668B5B;
    --cream: #F7F5EC;
    --tomato: #D94C43;
    --ink: #24372B;
}

.stApp {
    background: #F7F6F0;
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--forest) !important;
}

[data-testid="stSidebar"] {
    background: #EAF0E5;
    border-right: 1px solid #DCE5D7;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: var(--forest) !important;
}

.hero {
    background: linear-gradient(120deg, #244B35 0%, #3E7049 68%, #78966B 100%);
    padding: 38px 36px;
    border-radius: 24px;
    color: white;
    margin: 8px 0 24px 0;
}

.hero h1 {
    color: white !important;
    font-size: clamp(30px, 4vw, 48px);
    line-height: 1.15;
    margin-bottom: 12px;
}

.hero p {
    color: #F0F6E9;
    font-size: 16px;
    line-height: 1.7;
}

.eyebrow {
    color: #F8C7A6;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-size: 12px;
}

.section-label {
    color: #668B5B;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
}

.info-card {
    background: white;
    border: 1px solid #E5E9DF;
    border-radius: 18px;
    padding: 22px;
    min-height: 155px;
    margin-bottom: 12px;
    box-shadow: 0 5px 18px rgba(36, 75, 53, 0.04);
}

.info-card h3 {
    margin-top: 8px;
    font-size: 19px;
}

.info-card p {
    color: #647166;
    font-size: 14px;
    line-height: 1.65;
}

.tomato-tag {
    display: inline-block;
    background: #FBE7E2;
    color: #A73830;
    padding: 6px 12px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
}

.result-card {
    background: white;
    border: 1px solid #DFE7D9;
    border-radius: 18px;
    padding: 24px;
    margin: 12px 0;
}

div.stButton > button {
    border-radius: 12px;
    min-height: 44px;
    font-weight: 700;
    border: 1px solid #D6E1D1;
}

div.stButton > button[kind="primary"] {
    background: #D94C43;
    color: white;
    border: none;
}

div.stButton > button[kind="primary"]:hover {
    background: #B83B34;
    color: white;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E4E9DF;
    padding: 16px;
    border-radius: 15px;
}

hr {
    border-color: #E1E6DC;
}

.small-note {
    color: #6B776D;
    font-size: 12px;
    line-height: 1.6;
}

[data-testid="stFileUploader"] {
    background: white;
    padding: 14px;
    border-radius: 16px;
    border: 1px dashed #A8BDA0;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Preparing the AgriShield AI model..."):
            downloaded = gdown.download(
                MODEL_URL, MODEL_PATH, quiet=False
            )

        if not downloaded or not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "The model could not be downloaded. "
                "Check that the Google Drive file is shared with anyone "
                "who has the link."
            )

        if os.path.getsize(MODEL_PATH) < 1_000_000:
            os.remove(MODEL_PATH)
            raise RuntimeError(
                "The downloaded model file appears incomplete."
            )

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# =========================================================
# MODEL CLASS LABELS - ORDER MUST MATCH TRAINING
# =========================================================

class_names = [
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
    "Tomato___healthy",
]

TOMATO_START = 28
TOMATO_END = 38

# =========================================================
# TOMATO DISEASE INFORMATION
# =========================================================

disease_info = {
    "Tomato___Bacterial_spot": {
        "name": "Bacterial Spot",
        "emoji": "🟤",
        "summary": "A bacterial disease that can affect tomato leaves and fruit.",
        "symptoms": [
            "Small dark spots on leaves",
            "Yellow areas may develop around spots",
            "Fruit may develop raised or rough spots",
        ],
        "care": [
            "Avoid working with plants when foliage is wet.",
            "Remove badly affected plant material appropriately.",
            "Use clean tools and avoid splashing soil onto leaves.",
            "Ask a local agricultural expert about suitable control options.",
        ],
    },
    "Tomato___Early_blight": {
        "name": "Early Blight",
        "emoji": "🍂",
        "summary": "A fungal disease often associated with older leaves.",
        "symptoms": [
            "Brown spots with ring-like patterns",
            "Yellowing around affected areas",
            "Older leaves may dry and fall early",
        ],
        "care": [
            "Remove severely affected leaves when practical.",
            "Water near the soil instead of over the foliage.",
            "Keep plants appropriately spaced for airflow.",
            "Avoid moving infected plant debris between plants.",
        ],
    },
    "Tomato___Late_blight": {
        "name": "Late Blight",
        "emoji": "🌧️",
        "summary": "A potentially fast-spreading disease favoured by cool, wet conditions.",
        "symptoms": [
            "Irregular dark or water-soaked-looking patches",
            "Rapidly worsening leaf damage in favourable weather",
            "Fruit may develop firm, discoloured areas",
        ],
        "care": [
            "Seek prompt confirmation from an agricultural specialist.",
            "Avoid prolonged leaf wetness and improve airflow.",
            "Do not move potentially infected plants or debris around.",
            "Follow locally approved disease-management advice.",
        ],
    },
    "Tomato___Leaf_Mold": {
        "name": "Leaf Mold",
        "emoji": "🌿",
        "summary": "A fungal disease commonly favoured by high humidity.",
        "symptoms": [
            "Pale yellow patches on upper leaf surfaces",
            "Olive or greyish growth may appear underneath leaves",
            "Affected leaves can curl or dry",
        ],
        "care": [
            "Improve ventilation around plants.",
            "Water at soil level where possible.",
            "Avoid excessive humidity in protected growing areas.",
            "Remove affected material carefully.",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "emoji": "🔎",
        "summary": "A fungal leaf disease that often starts on older foliage.",
        "symptoms": [
            "Many small, circular spots",
            "Spots may have pale centres and dark edges",
            "Lower leaves may yellow and dry",
        ],
        "care": [
            "Remove heavily affected lower leaves.",
            "Keep foliage as dry as practical.",
            "Clear infected debris from the growing area.",
            "Use mulch to reduce soil splash where suitable.",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Two-Spotted Spider Mites",
        "emoji": "🕷️",
        "summary": "Tiny pests that feed on leaves and can cause speckled damage.",
        "symptoms": [
            "Fine pale speckles on leaves",
            "Yellowing or a bronze appearance",
            "Fine webbing may appear during heavier infestations",
        ],
        "care": [
            "Inspect the undersides of leaves carefully.",
            "Check nearby plants for similar signs.",
            "Protect beneficial insects where possible.",
            "Ask a local expert about appropriate pest management.",
        ],
    },
    "Tomato___Target_Spot": {
        "name": "Target Spot",
        "emoji": "🎯",
        "summary": "A leaf disease that may produce brown spots with concentric rings.",
        "symptoms": [
            "Circular brown leaf spots",
            "Ring patterns in some lesions",
            "Affected leaves may yellow or drop",
        ],
        "care": [
            "Improve airflow by maintaining suitable plant spacing.",
            "Avoid unnecessary wetting of foliage.",
            "Remove badly affected plant debris.",
            "Confirm the disease before choosing a treatment.",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "emoji": "🍃",
        "summary": "A viral disease commonly spread by whiteflies.",
        "symptoms": [
            "Leaves may curl upward",
            "Leaves can become yellow and smaller",
            "Plants may show stunted growth",
        ],
        "care": [
            "Monitor plants for whiteflies.",
            "Remove confirmed infected plants where recommended locally.",
            "Use locally recommended measures to manage whiteflies.",
            "Choose suitable resistant varieties when planning future crops.",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "emoji": "🧩",
        "summary": "A viral disease that can cause uneven leaf colouring and distortion.",
        "symptoms": [
            "Mottled light and dark green leaf patterns",
            "Leaves may become distorted",
            "Plant growth may be reduced",
        ],
        "care": [
            "Clean hands and tools after handling suspect plants.",
            "Avoid handling healthy plants immediately after affected plants.",
            "Remove and dispose of confirmed infected material according to local guidance.",
            "Use clean seed and healthy seedlings.",
        ],
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "emoji": "🌱",
        "summary": "The model classified the image as a healthy tomato leaf.",
        "symptoms": [
            "No target disease was identified by the model.",
            "Leaves may appear green and reasonably uniform.",
            "Some real problems may not be visible in the uploaded image.",
        ],
        "care": [
            "Water consistently according to soil and weather conditions.",
            "Monitor the undersides of leaves regularly.",
            "Maintain good airflow around plants.",
            "Continue checking for changes in leaves, stems and fruit.",
        ],
    },
}


def format_label(label):
    return label.replace("Tomato___", "").replace("_", " ").strip()


def predict_tomato(image_bytes):
    model = load_model()

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))

    array = tf.keras.preprocessing.image.img_to_array(image)
    array = np.expand_dims(array, axis=0)

    predictions = model.predict(array, verbose=0)[0]

    if len(predictions) != len(class_names):
        raise ValueError(
            "The model output does not match the configured class list."
        )

    # Use the model's original full 38-class prediction.
    overall_index = int(np.argmax(predictions))
    overall_label = class_names[overall_index]

    # Do not relabel a non-tomato prediction as a tomato disease.
    if not overall_label.startswith("Tomato___"):
        return {
            "is_tomato": False,
            "label": overall_label,
            "confidence": float(predictions[overall_index]) * 100,
        }

    return {
        "is_tomato": True,
        "label": overall_label,
        "confidence": float(predictions[overall_index]) * 100,
    }


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("# 🍅 AgriShield")
    st.caption("TOMATO HEALTH ASSISTANT")
    st.markdown("---")

    page = st.radio(
        "EXPLORE",
        [
            "Home",
            "Disease Detection",
            "Tomato Care",
            "Disease Library",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🌱 Our focus")
    st.write("Helping users learn about tomato leaf health.")
    st.caption("AI predictions are estimates and should be verified.")

# =========================================================
# HOME
# =========================================================

if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">SMARTER TOMATO CARE</div>
        <h1>Healthy leaves.<br>Stronger tomato plants.</h1>
        <p>
            Meet AgriShield — a tomato-focused disease recognition
            and plant-care guide designed to help you understand
            leaf symptoms and learn about prevention.
        </p>
        <span class="tomato-tag">🍅 MADE FOR TOMATO PLANTS</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Welcome to your tomato care space")
    st.write(
        "Explore possible tomato leaf diseases, learn everyday care "
        "practices and discover ways to monitor plant health."
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("""
        <div class="info-card">
            <div class="section-label">01 / IDENTIFY</div>
            <h3>🔬 Tomato Disease Detection</h3>
            <p>
                Upload a clear tomato leaf photo to see the model's
                predicted category, confidence score and related
                educational guidance.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🍅  Check a Tomato Leaf", type="primary",
                     use_container_width=True):
            st.session_state["page_choice"] = "Disease Detection"
            st.rerun()

    with right:
        st.markdown("""
        <div class="info-card">
            <div class="section-label">02 / PREVENT</div>
            <h3>🌿 Better Tomato Care</h3>
            <p>
                Learn about watering, soil, sunlight, airflow and
                simple habits that help maintain healthier plants.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Explore Tomato Care", use_container_width=True):
            st.session_state["page_choice"] = "Tomato Care"
            st.rerun()

    # Apply navigation requested by home-page buttons.
    if "page_choice" in st.session_state:
        choice = st.session_state.pop("page_choice")
        st.session_state["_navigation"] = choice
        st.rerun()

    st.markdown("### What you can explore")

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="info-card">
            <h3>🍃 Recognise</h3>
            <p>Learn about the tomato leaf categories covered by the AI model.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="info-card">
            <h3>💧 Maintain</h3>
            <p>Understand practical watering, soil and plant-monitoring habits.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="info-card">
            <h3>🛡️ Prevent</h3>
            <p>Explore general prevention practices and when to seek expert advice.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p class="small-note">AgriShield is an educational project. '
        'Its image classifier is not a substitute for an agricultural '
        'diagnosis.</p>',
        unsafe_allow_html=True,
    )

# =========================================================
# DISEASE DETECTION
# =========================================================

elif page == "Disease Detection":
    st.markdown('<div class="section-label">TOMATO HEALTH SCANNER</div>',
                unsafe_allow_html=True)
    st.title("🍅 Tomato Disease Detection")
    st.write(
        "Upload a clear image of a tomato leaf. For the best result, "
        "keep the leaf in focus and use good lighting."
    )

    uploaded = st.file_uploader(
        "Choose a tomato leaf image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear, close-up image with as little background as possible.",
    )

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        preview = Image.open(BytesIO(image_bytes)).convert("RGB")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.image(preview, caption="Your uploaded image",
                     use_container_width=True)

        with right:
            st.markdown("### Ready to analyse?")
            st.write(
                "The model compares the image with the leaf categories "
                "it learned during training."
            )

            if st.button("🔍 Analyse Tomato Leaf", type="primary",
                         use_container_width=True):
                try:
                    with st.spinner("Analysing leaf patterns..."):
                        result = predict_tomato(image_bytes)

                    st.session_state["prediction_result"] = result

                except Exception as error:
                    st.error("The prediction could not be completed.")
                    st.caption(f"Technical details: {error}")

    result = st.session_state.get("prediction_result")

    if uploaded is not None and result is not None:
        st.markdown("---")
        st.markdown("## Your result")

        if not result["is_tomato"]:
            st.warning(
                "The model's highest-scoring category is not a tomato "
                "category. Please upload a clear tomato leaf image. "
                "This result does not prove that the image belongs to "
                "another crop; the model can make mistakes."
            )
            st.caption(
                "Highest-scoring model category: "
                + format_label(result["label"])
                + f" ({result['confidence']:.2f}% model score)"
            )

        else:
            label = result["label"]
            details = disease_info.get(label)

            if details:
                if label == "Tomato___healthy":
                    st.success("🌱 " + details["name"])
                else:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="section-label">MODEL PREDICTION</div>'
                        f'<h2>{details["emoji"]} {details["name"]}</h2>'
                        f'<p>{details["summary"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.metric("Model confidence score",
                          f"{result['confidence']:.2f}%")

                st.caption(
                    "This score reflects the model's output, not a "
                    "guaranteed probability that the diagnosis is correct."
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🔎 Signs to look for")
                    for item in details["symptoms"]:
                        st.markdown(f"- {item}")

                with col2:
                    st.markdown("### 🛡️ Suggested next steps")
                    for item in details["care"]:
                        st.markdown(f"- {item}")

                st.info(
                    "Compare these notes with the actual plant. Similar "
                    "symptoms can have different causes. Confirm the "
                    "problem with a qualified agricultural specialist "
                    "before applying treatments."
                )

            if st.button("↻ Analyse another image"):
                st.session_state.pop("prediction_result", None)
                st.rerun()

    elif uploaded is None:
        st.markdown("""
        <div class="info-card">
            <h3>📷 Tips for a useful image</h3>
            <p>• Photograph a single leaf clearly.</p>
            <p>• Use natural light when possible.</p>
            <p>• Avoid blurry images and strong shadows.</p>
            <p>• The tool works best with tomato leaf images similar
            to the categories in its training dataset.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TOMATO CARE
# =========================================================

elif page == "Tomato Care":
    st.markdown('<div class="section-label">GROW WITH CONFIDENCE</div>',
                unsafe_allow_html=True)
    st.title("🌿 Tomato Care Guide")
    st.write(
        "Simple growing practices can help keep tomato plants healthy "
        "and make it easier to notice problems early."
    )

    a, b = st.columns(2, gap="large")

    with a:
        st.markdown("""
        <div class="info-card">
            <h3>☀️ Sunlight</h3>
            <p>Tomatoes generally grow best in a sunny location with
            several hours of direct sunlight. Adapt care to local
            weather and growing conditions.</p>
        </div>
        <div class="info-card">
            <h3>💧 Watering</h3>
            <p>Water near the soil and aim for consistent moisture.
            Avoid prolonged waterlogging and unnecessary wetting
            of the leaves.</p>
        </div>
        <div class="info-card">
            <h3>🌱 Soil and drainage</h3>
            <p>Use fertile, well-drained soil. Good drainage helps
            protect roots from problems associated with excess water.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="info-card">
            <h3>🌬️ Airflow and spacing</h3>
            <p>Give plants suitable space for air to move between
            leaves. Avoid overcrowding and follow local growing advice.</p>
        </div>
        <div class="info-card">
            <h3>🔎 Regular inspection</h3>
            <p>Check the upper and lower sides of leaves, stems and
            fruit. Notice changes early and record how they develop.</p>
        </div>
        <div class="info-card">
            <h3>🧹 Garden hygiene</h3>
            <p>Clean tools as appropriate and handle suspected infected
            plant material carefully to reduce the chance of spreading
            some diseases.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Weekly tomato plant check")

    checks = [
        "Inspect the top and underside of leaves",
        "Check soil moisture and drainage",
        "Look for insects or fine webbing",
        "Check stems and fruit for unusual spots",
        "Remove fallen plant debris where appropriate",
    ]

    for check in checks:
        st.checkbox(check, key="care_" + check)

    st.info(
        "Fertilizer needs vary with soil, plant stage and growing "
        "conditions. Avoid applying fertilizer or pesticides solely "
        "because of an AI prediction; use local agricultural guidance."
    )

# =========================================================
# DISEASE LIBRARY
# =========================================================

elif page == "Disease Library":
    st.markdown('<div class="section-label">LEARN THE SIGNS</div>',
                unsafe_allow_html=True)
    st.title("🍃 Tomato Disease Library")
    st.write(
        "Explore the tomato categories recognised by the current model. "
        "Descriptions are general educational guidance."
    )

    search = st.text_input(
        "Search tomato diseases",
        placeholder="Try early blight or leaf mold...",
    )

    for label in class_names[TOMATO_START:TOMATO_END]:
        details = disease_info[label]

        if search and search.lower() not in (
            details["name"] + " " + details["summary"]
        ).lower():
            continue

        with st.expander(f"{details['emoji']}  {details['name']}"):
            st.write(details["summary"])
            st.markdown("**Common signs**")
            for item in details["symptoms"]:
                st.markdown(f"- {item}")

            st.markdown("**General care and prevention**")
            for item in details["care"]:
                st.markdown(f"- {item}")

    st.caption(
        "A visual match is not definitive diagnosis. Some diseases "
        "share similar symptoms."
    )

# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "About Project":
    st.markdown('<div class="section-label">THE PROJECT</div>',
                unsafe_allow_html=True)
    st.title("About AgriShield")
    st.write(
        "AgriShield is a tomato-focused educational web application "
        "that uses a trained image-classification model to estimate "
        "the category of an uploaded leaf image."
    )

    a, b, c = st.columns(3)

    with a:
        st.metric("Model output classes", "38")
    with b:
        st.metric("Tomato categories", "10")
    with c:
        st.metric("Image input", "128 × 128")

    st.markdown("### How the system works")

    steps = [
        ("01", "Upload", "The user submits a tomato leaf image."),
        ("02", "Preprocess", "The image is resized to 128 × 128 pixels."),
        ("03", "Classify", "The CNN model returns scores for its 38 classes."),
        ("04", "Present", "The website displays tomato-specific information when the top class is a tomato category."),
    ]

    for number, title, description in steps:
        st.markdown(
            f'<div class="info-card">'
            f'<span class="tomato-tag">STEP {number}</span>'
            f'<h3>{title}</h3><p>{description}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Technologies used")
    st.markdown(
        "- **Python** — application logic\n"
        "- **Streamlit** — interactive web interface\n"
        "- **TensorFlow / Keras** — trained image-classification model\n"
        "- **CNN** — image feature learning and classification\n"
        "- **NumPy and Pillow** — image preparation and array processing"
    )

    st.markdown("### Important limitation")
    st.write(
        "The model was trained on a fixed dataset. Its output can be "
        "wrong on field images, unfamiliar lighting or other conditions. "
        "The application is a learning and decision-support aid, not a "
        "replacement for laboratory testing or expert agricultural advice."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    '<p class="small-note" style="text-align:center;">'
    '🍅 AgriShield · Tomato Health Assistant · '
    'Learn, observe and care for your plants.'
    '</p>',
    unsafe_allow_html=True,
)
