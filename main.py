
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
    page_title="AgriShield | Tomato Health Assistant",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------
MODEL_PATH = "trained_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1YNIs92QlZIJP0YQ2e5jGWUAuEWgFR_lE"

# Keep this class order exactly as used during training.
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
    "Tomato___healthy",
]

TOMATO_CLASSES = CLASS_NAMES[28:38]

# --------------------------------------------------
# TOMATO DISEASE INFORMATION
# --------------------------------------------------
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "name": "Bacterial Spot",
        "type": "Bacterial disease",
        "symptoms": "Small dark spots may appear on leaves, sometimes with yellow areas around them.",
        "care": [
            "Avoid wetting the leaves while watering.",
            "Remove badly affected plant material safely.",
            "Keep plants spaced to improve airflow.",
            "Ask a local agricultural expert about suitable management."
        ]
    },
    "Tomato___Early_blight": {
        "name": "Early Blight",
        "type": "Fungal disease",
        "symptoms": "Brown spots with ring-like patterns may develop, often on older leaves.",
        "care": [
            "Remove heavily affected leaves when appropriate.",
            "Water near the soil instead of over the leaves.",
            "Use mulch to reduce soil splashing.",
            "Follow local agricultural guidance for disease management."
        ]
    },
    "Tomato___Late_blight": {
        "name": "Late Blight",
        "type": "Disease caused by an oomycete",
        "symptoms": "Irregular, water-soaked-looking patches may spread quickly in cool, damp conditions.",
        "care": [
            "Check nearby plants for similar symptoms.",
            "Avoid overhead watering and prolonged leaf wetness.",
            "Seek prompt advice from an agricultural specialist.",
            "Follow local guidance on appropriate control measures."
        ]
    },
    "Tomato___Leaf_Mold": {
        "name": "Leaf Mold",
        "type": "Fungal disease",
        "symptoms": "Pale yellow patches may appear on the upper leaf surface, with mold-like growth underneath.",
        "care": [
            "Improve airflow around plants.",
            "Avoid excessive humidity and leaf wetness.",
            "Remove affected plant material when appropriate.",
            "Keep growing areas clean."
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "type": "Fungal disease",
        "symptoms": "Many small, round spots with pale centers and darker edges may appear on leaves.",
        "care": [
            "Remove badly affected lower leaves when appropriate.",
            "Avoid splashing soil onto foliage.",
            "Water at the base of the plant.",
            "Clean tools after working with affected plants."
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Two-Spotted Spider Mites",
        "type": "Pest damage",
        "symptoms": "Fine pale speckles, yellowing and sometimes delicate webbing may appear on leaves.",
        "care": [
            "Inspect the undersides of leaves closely.",
            "Check nearby plants for similar signs.",
            "Avoid unnecessary broad-spectrum pesticide use.",
            "Ask an agricultural expert about appropriate mite management."
        ]
    },
    "Tomato___Target_Spot": {
        "name": "Target Spot",
        "type": "Fungal disease",
        "symptoms": "Brown lesions with concentric rings may appear on leaves and sometimes other plant parts.",
        "care": [
            "Improve airflow around the plant.",
            "Keep foliage as dry as practical.",
            "Remove affected plant material when appropriate.",
            "Seek local advice if symptoms continue to spread."
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "type": "Viral disease",
        "symptoms": "Leaves may curl upward, turn yellow and become smaller, while plant growth may be reduced.",
        "care": [
            "Inspect plants for whiteflies, which can spread the virus.",
            "Remove severely affected plants when advised by a local expert.",
            "Use locally recommended whitefly prevention methods.",
            "Choose suitable resistant varieties when planning future crops."
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "type": "Viral disease",
        "symptoms": "Leaves may show mottled light and dark green areas, with distorted growth in some cases.",
        "care": [
            "Wash hands before handling healthy plants.",
            "Clean gardening tools between plants.",
            "Avoid handling plants when foliage is wet.",
            "Seek expert advice about managing suspected viral infections."
        ]
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "type": "Healthy category",
        "symptoms": "The model identifies this image as belonging to the healthy tomato-leaf category.",
        "care": [
            "Water consistently according to soil and weather conditions.",
            "Provide adequate sunlight and airflow.",
            "Inspect leaves regularly for changes.",
            "Maintain balanced soil nutrition."
        ]
    }
}

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F8F7F1;
        color: #183D2B;
    }

    [data-testid="stSidebar"] {
        background-color: #EAF0E5;
        border-right: 1px solid #D9E1D4;
    }

    h1, h2, h3 {
        color: #174C36 !important;
    }

    .hero {
        background: linear-gradient(115deg, #194F38, #78966C);
        padding: 32px;
        border-radius: 22px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1, .hero h2, .hero p {
        color: white !important;
    }

    .eyebrow {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #D8E7C8;
        text-transform: uppercase;
    }

    .info-card {
        background: white;
        border: 1px solid #DFE6DA;
        border-radius: 18px;
        padding: 23px;
        min-height: 150px;
        margin-bottom: 12px;
    }

    .info-card h3 {
        font-size: 20px;
        margin-top: 8px;
    }

    .muted {
        color: #68766C;
        font-size: 14px;
    }

    .result-card {
        background: #FFFFFF;
        border: 1px solid #DCE6D8;
        border-radius: 18px;
        padding: 25px;
        margin-top: 12px;
        margin-bottom: 15px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
        border: 1px solid #D5E0D0;
    }

    div.stButton > button[kind="primary"] {
        background-color: #D94D43;
        color: white;
        border: none;
    }

    .footer {
        border-top: 1px solid #DCE2D8;
        margin-top: 35px;
        padding-top: 16px;
        color: #748075;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
PAGES = [
    "Home",
    "Disease Detection",
    "Tomato Care",
    "Disease Library",
    "About Project"
]


def go_to_page(target):
    """Set the sidebar page before Streamlit reruns the script."""
    st.session_state["page_choice"] = target


if "page_choice" not in st.session_state:
    st.session_state["page_choice"] = "Home"

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

if "prediction_image" not in st.session_state:
    st.session_state["prediction_image"] = None

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        "<h2>🍅 AgriShield</h2>"
        "<p style='color:#78847B;font-size:12px;letter-spacing:1px;'>"
        "TOMATO HEALTH ASSISTANT</p>",
        unsafe_allow_html=True
    )

    st.divider()

    # IMPORTANT: Both home buttons update this same widget state.
    st.radio(
        "Navigate",
        PAGES,
        key="page_choice",
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 🌱 Our focus")
    st.write("Helping users learn about tomato leaf health.")

    st.caption(
        "AI predictions are estimates, not confirmed diagnoses. "
        "Verify important decisions with an agricultural expert."
    )

page = st.session_state["page_choice"]

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading the AgriShield model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(
            MODEL_URL,
            MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "The model file could not be downloaded. "
            "Please check the Google Drive sharing settings."
        )

    if os.path.getsize(MODEL_PATH) == 0:
        raise ValueError("The downloaded model file is empty.")

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


def predict_leaf(image):
    image = image.convert("RGB").resize((128, 128))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    model = load_model()
    probabilities = model.predict(array, verbose=0)[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"The model returned {len(probabilities)} classes, "
            f"but {len(CLASS_NAMES)} labels are configured."
        )

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    return predicted_index, confidence


def show_footer():
    st.markdown(
        "<div class='footer'>AgriShield · Tomato Leaf Health Education"
        "<br>AI predictions are informational and should be verified.</div>",
        unsafe_allow_html=True
    )


# ==================================================
# PAGE 1: HOME
# ==================================================
if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">SMARTER TOMATO CARE</div>
        <h1>Grow healthier tomatoes with AgriShield 🍅</h1>
        <p>Explore tomato leaf health, learn about common diseases,
        and discover practical plant-care guidance.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Welcome to your tomato care space")
    st.write(
        "Explore possible tomato leaf diseases, learn everyday care "
        "practices and discover ways to monitor plant health."
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("""
        <div class="info-card">
            <p class="eyebrow" style="color:#527B50">01 / IDENTIFY</p>
            <h3>🔬 Tomato Disease Detection</h3>
            <p>Upload a clear tomato leaf photo to view the model's "
            "predicted category, confidence score and educational guidance.</p>
        </div>
        """, unsafe_allow_html=True)

        # FIXED: updates the sidebar radio selection.
        st.button(
            "🍅 Check a Tomato Leaf",
            key="home_detect",
            type="primary",
            use_container_width=True,
            on_click=go_to_page,
            args=("Disease Detection",)
        )

    with right:
        st.markdown("""
        <div class="info-card">
            <p class="eyebrow" style="color:#527B50">02 / PREVENT</p>
            <h3>🌿 Better Tomato Care</h3>
            <p>Learn about watering, soil, sunlight, airflow and simple
            habits that help maintain healthier plants.</p>
        </div>
        """, unsafe_allow_html=True)

        # FIXED: updates the sidebar radio selection.
        st.button(
            "Explore Tomato Care",
            key="home_care",
            use_container_width=True,
            on_click=go_to_page,
            args=("Tomato Care",)
        )

    st.markdown("## What you can explore")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>🍃 Recognise</h3>
            <p>Learn about tomato leaf categories and common symptoms.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Open Disease Library",
            key="home_library",
            use_container_width=True,
            on_click=go_to_page,
            args=("Disease Library",)
        )

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>💧 Maintain</h3>
            <p>Understand practical watering, soil and plant-care habits.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "View Care Tips",
            key="home_care2",
            use_container_width=True,
            on_click=go_to_page,
            args=("Tomato Care",)
        )

    with c3:
        st.markdown("""
        <div class="info-card">
            <h3>🛡️ Learn</h3>
            <p>Explore preventive practices and learn about the project.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "About AgriShield",
            key="home_about",
            use_container_width=True,
            on_click=go_to_page,
            args=("About Project",)
        )

    show_footer()


# ==================================================
# PAGE 2: DISEASE DETECTION
# ==================================================
elif page == "Disease Detection":
    st.markdown("## 🍅 Tomato Disease Detection")
    st.write(
        "Upload a clear image of a tomato leaf. AgriShield will compare "
        "the image with the categories learned by its trained model."
    )

    st.info(
        "For best results, use a clear, well-lit image with the leaf "
        "visible and in focus. The tool is designed for tomato leaves."
    )

    uploaded_file = st.file_uploader(
        "Upload tomato leaf image",
        type=["jpg", "jpeg", "png"],
        key="tomato_upload"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            left, right = st.columns([1, 1], gap="large")

            with left:
                st.image(
                    image,
                    caption="Uploaded leaf image",
                    use_container_width=True
                )

            with right:
                st.markdown("### Ready to check?")
                st.write(
                    "Select the button below to run the model on your image."
                )

                if st.button(
                    "🔍 Analyse Tomato Leaf",
                    type="primary",
                    use_container_width=True,
                    key="analyse_leaf"
                ):
                    try:
                        with st.spinner("Analysing the image..."):
                            index, confidence = predict_leaf(image)

                        predicted_class = CLASS_NAMES[index]

                        # Keep the interface tomato-only.
                        if not predicted_class.startswith("Tomato___"):
                            st.session_state["prediction_result"] = {
                                "status": "not_tomato",
                                "class": predicted_class,
                                "confidence": confidence,
                                "image_name": uploaded_file.name
                            }
                        else:
                            st.session_state["prediction_result"] = {
                                "status": "tomato",
                                "class": predicted_class,
                                "confidence": confidence,
                                "image_name": uploaded_file.name
                            }

                        st.session_state["prediction_image"] = uploaded_file.name

                    except Exception as e:
                        st.error(
                            "The model could not analyse this image. "
                            "Please try again."
                        )
                        st.caption(f"Technical details: {e}")

        except Exception:
            st.error("This file could not be opened as an image.")

    result = st.session_state.get("prediction_result")

    # Only show results for the currently uploaded file.
    if (
        uploaded_file is not None
        and result is not None
        and result.get("image_name") == uploaded_file.name
    ):
        st.divider()

        if result["status"] == "not_tomato":
            st.warning(
                "The model's top prediction is not a tomato-leaf category. "
                "Please upload a clear tomato leaf image."
            )

        else:
            info = DISEASE_INFO[result["class"]]

            st.markdown("## Analysis result")

            st.markdown(
                f"""
                <div class="result-card">
                    <p class="eyebrow" style="color:#527B50">PREDICTED CATEGORY</p>
                    <h2>{info['name']}</h2>
                    <p class="muted">{info['type']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Model confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.progress(
                min(max(result["confidence"], 0.0), 1.0)
            )

            st.caption(
                "Confidence is the model's score for its predicted class. "
                "It does not guarantee that the prediction is correct."
            )

            st.markdown("### 🔎 What to look for")
            st.write(info["symptoms"])

            st.markdown("### 🌱 Suggested care and next steps")
            for item in info["care"]:
                st.markdown(f"- {item}")

            if result["class"] != "Tomato___healthy":
                st.warning(
                    "This is an AI-generated prediction, not a confirmed "
                    "diagnosis. Similar symptoms can have different causes. "
                    "Consult an agricultural expert before using treatments."
                )
            else:
                st.success(
                    "The model classified this image as a healthy tomato "
                    "leaf. Continue checking the plant regularly."
                )

    show_footer()


# ==================================================
# PAGE 3: TOMATO CARE
# ==================================================
elif page == "Tomato Care":
    st.markdown("## 🌿 Better Tomato Care")
    st.write(
        "Simple everyday practices can help tomato plants grow well "
        "and make it easier to notice problems early."
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>💧 Watering</h3>
            Water consistently according to the weather and soil moisture.
            Prefer watering near the soil and avoid keeping the soil
            constantly waterlogged.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>☀️ Sunlight</h3>
            Tomatoes generally grow best with plenty of sunlight.
            Choose a suitable growing location and monitor plants during
            very hot weather.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>🌱 Soil and Nutrition</h3>
            Use well-drained soil and balanced nutrition. Avoid applying
            fertiliser without considering the plant's needs and local advice.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>🌬️ Airflow</h3>
            Provide adequate spacing between plants. Good airflow helps
            leaves dry and can reduce conditions favourable to some diseases.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>🔍 Regular Inspection</h3>
            Check both sides of leaves for spots, yellowing, curling,
            unusual patterns or pests. Notice whether symptoms are spreading.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>🧹 Garden Hygiene</h3>
            Keep the growing area clean and clean tools between plants.
            Handle suspected diseased plants carefully.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Daily tomato plant checklist")

    checklist = [
        "Check leaves for unusual spots or colour changes",
        "Check soil moisture before watering",
        "Look for pests under the leaves",
        "Check that plants have enough airflow",
        "Record any symptoms that are getting worse"
    ]

    for item in checklist:
        st.checkbox(item, key=f"care_{item}")

    st.caption(
        "These are general educational tips. Specific care depends on "
        "the tomato variety, growing conditions and local climate."
    )

    show_footer()


# ==================================================
# PAGE 4: DISEASE LIBRARY
# ==================================================
elif page == "Disease Library":
    st.markdown("## 📚 Tomato Disease Library")
    st.write(
        "Browse the tomato categories supported by the trained model. "
        "The symptoms listed are common signs, not a definitive diagnosis."
    )

    names = {
        key: value["name"]
        for key, value in DISEASE_INFO.items()
    }

    selected_name = st.selectbox(
        "Choose a tomato category",
        options=list(names.keys()),
        format_func=lambda key: names[key]
    )

    info = DISEASE_INFO[selected_name]

    st.markdown(
        f"""
        <div class="result-card">
            <p class="eyebrow" style="color:#527B50">TOMATO HEALTH GUIDE</p>
            <h2>{info['name']}</h2>
            <p class="muted">{info['type']}</p>
            <h3>Common symptoms</h3>
            <p>{info['symptoms']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Suggested care and next steps")
    for item in info["care"]:
        st.markdown(f"- {item}")

    st.divider()
    st.markdown("### Categories supported by AgriShield")

    for key, value in DISEASE_INFO.items():
        st.markdown(f"- **{value['name']}** — {value['type']}")

    show_footer()


# ==================================================
# PAGE 5: ABOUT PROJECT
# ==================================================
elif page == "About Project":
    st.markdown("## 🌱 About AgriShield")

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">OUR COLLEGE PROJECT</div>
        <h2>AI-assisted tomato leaf health education</h2>
        <p>
            AgriShield is a plant-disease identification project that uses
            a trained deep-learning image classification model to estimate
            the category of a submitted leaf image.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Project objectives")

    st.markdown("""
    - Explore image-based tomato leaf classification.
    - Present model predictions in a simple website.
    - Share educational information about common tomato leaf problems.
    - Encourage regular plant inspection and informed prevention practices.
    """)

    st.markdown("### ⚙️ How the system works")

    steps = [
        ("01", "Image Upload", "The user uploads a tomato leaf image."),
        ("02", "Preprocessing", "The image is resized to 128 × 128 pixels and normalised."),
        ("03", "CNN Model", "The trained model evaluates the image and predicts a class."),
        ("04", "Result", "The website displays the predicted category and confidence score."),
        ("05", "Guidance", "The user sees educational symptoms and general care suggestions.")
    ]

    for number, title, description in steps:
        st.markdown(
            f"""
            <div class="info-card" style="min-height:90px;">
                <p class="eyebrow" style="color:#527B50">{number}</p>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🧰 Technologies used")

    tech1, tech2 = st.columns(2)

    with tech1:
        st.markdown("""
        - **Python:** Main programming language
        - **TensorFlow / Keras:** Loads and runs the trained model
        - **CNN:** Learns visual patterns in leaf images
        """)

    with tech2:
        st.markdown("""
        - **Streamlit:** Website interface
        - **NumPy:** Image array processing
        - **Pillow:** Image opening and resizing
        """)

    st.markdown("### ⚠️ Limitations")

    st.markdown("""
    - Predictions depend on image quality and the training data.
    - Similar symptoms can arise from different causes.
    - A high confidence score does not guarantee correctness.
    - The website does not replace laboratory testing or professional advice.
    - Treatment decisions should be checked with a qualified agricultural expert.
    """)

    show_footer()
