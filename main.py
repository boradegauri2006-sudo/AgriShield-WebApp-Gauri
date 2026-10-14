import os
import io
import json
import datetime
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================================================
# AGRISHIELD — CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AgriShield | Tomato Health",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "trained_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1YNIs92QlZIJP0YQ2e5jGWUAuEWgFR_lE"
HISTORY_PATH = "diagnosis_history.json"
IMAGE_SIZE = (128, 128)

# Disease photos should be stored in the repository's assets/ folder.
DISEASE_IMAGES = {
    "Tomato___Bacterial_spot": "assets/bacterial_spot.jpg",
    "Tomato___Early_blight": "assets/early_blight.jpg",
    "Tomato___Late_blight": "assets/late_blight.jpg",
    "Tomato___Leaf_Mold": "assets/leaf_mold.jpg",
    "Tomato___Septoria_leaf_spot": "assets/septoria_leaf_spot.jpg",
    "Tomato___Spider_mites Two-spotted_spider_mite": "assets/spider_mites.jpg",
    "Tomato___Target_Spot": "assets/target_spot.jpg",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "assets/yellow_leaf_curl.jpg",
    "Tomato___Tomato_mosaic_virus": "assets/mosaic_virus.jpg",
    "Tomato___healthy": "assets/healthy_leaf.jpg",
}

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

TOMATO_INDICES = [
    i for i, name in enumerate(CLASS_NAMES)
    if name.startswith("Tomato")
]

# =========================================================
# ENGLISH + MARATHI CONTENT
# =========================================================
DISEASES = {
    "Tomato___Bacterial_spot": {
        "en": {
            "name": "Bacterial Spot",
            "kind": "Bacterial",
            "urgency": "Medium",
            "cause": "Bacteria",
            "summary": "Small dark spots may appear on leaves and fruit.",
            "symptoms": [
                "Small dark brown or black spots",
                "Possible yellow areas around spots",
                "Rough, scab-like marks on green fruit",
            ],
            "actions": [
                "Remove badly affected leaves carefully.",
                "Avoid overhead watering and working with wet plants.",
                "Ask a local agriculture expert whether a suitable treatment is needed.",
            ],
            "prevention": [
                "Use healthy seeds and seedlings.",
                "Water near the base of the plant.",
                "Rotate crops and remove infected plant debris.",
            ],
        },
        "mr": {
            "name": "जिवाणूजन्य ठिपके",
            "kind": "Bacterial",
            "urgency": "Medium",
            "cause": "जिवाणू",
            "summary": "पानांवर आणि फळांवर लहान काळसर ठिपके दिसू शकतात.",
            "symptoms": [
                "पानांवर लहान तपकिरी किंवा काळे ठिपके",
                "ठिपक्यांभोवती पिवळसर भाग दिसू शकतो",
                "हिरव्या फळांवर खरखरीत डाग",
            ],
            "actions": [
                "जास्त बाधित पाने काळजीपूर्वक काढा.",
                "पानांवर पाणी शिंपडणे टाळा.",
                "योग्य उपचारासाठी स्थानिक कृषी तज्ज्ञांचा सल्ला घ्या.",
            ],
            "prevention": [
                "निरोगी बियाणे आणि रोपे वापरा.",
                "झाडाच्या मुळाजवळ पाणी द्या.",
                "पीक फेरपालट करा आणि बाधित अवशेष काढा.",
            ],
        },
    },
    "Tomato___Early_blight": {
        "en": {
            "name": "Early Blight",
            "kind": "Fungal",
            "urgency": "Medium",
            "cause": "Fungus",
            "summary": "Brown spots with ring-like patterns, often on older lower leaves.",
            "symptoms": [
                "Brown circular spots with concentric rings",
                "Yellowing around the spots",
                "Older lower leaves may dry and fall",
            ],
            "actions": [
                "Remove badly affected lower leaves.",
                "Keep foliage dry and improve airflow.",
                "Seek local advice if the disease continues to spread.",
            ],
            "prevention": [
                "Rotate crops.",
                "Remove old infected plant material.",
                "Use mulch and water at the base.",
            ],
        },
        "mr": {
            "name": "करपा (अर्ली ब्लाइट)",
            "kind": "Fungal",
            "urgency": "Medium",
            "cause": "बुरशी",
            "summary": "विशेषतः खालच्या जुन्या पानांवर गोलाकार तपकिरी डाग दिसतात.",
            "symptoms": [
                "वलयांसारखे गोल तपकिरी डाग",
                "डागांभोवती पिवळसरपणा",
                "खालची पाने वाळून गळू शकतात",
            ],
            "actions": [
                "जास्त बाधित खालची पाने काढा.",
                "पाने शक्यतो कोरडी ठेवा आणि हवा खेळती ठेवा.",
                "रोग पसरत असल्यास कृषी तज्ज्ञांचा सल्ला घ्या.",
            ],
            "prevention": [
                "पीक फेरपालट करा.",
                "जुने बाधित अवशेष काढून टाका.",
                "आच्छादन वापरा आणि मुळाजवळ पाणी द्या.",
            ],
        },
    },
    "Tomato___Late_blight": {
        "en": {
            "name": "Late Blight",
            "kind": "Fungal-like",
            "urgency": "High",
            "cause": "Water mold",
            "summary": "Large water-soaked patches can spread rapidly in cool, wet conditions.",
            "symptoms": [
                "Large irregular dark patches",
                "Possible pale growth under leaves in humid weather",
                "Dark patches on stems or fruit",
            ],
            "actions": [
                "Act promptly and separate badly affected plant material.",
                "Avoid moving wet plant material between plants.",
                "Contact an agriculture expert quickly because it can spread rapidly.",
            ],
            "prevention": [
                "Avoid wetting foliage.",
                "Allow good spacing and airflow.",
                "Monitor tomato and potato plants regularly.",
            ],
        },
        "mr": {
            "name": "उशिराचा करपा",
            "kind": "Fungal-like",
            "urgency": "High",
            "cause": "बुरशीसारखा रोगकारक",
            "summary": "थंड आणि दमट हवेत मोठे पाणचट डाग वेगाने पसरू शकतात.",
            "symptoms": [
                "मोठे अनियमित काळसर डाग",
                "दमट हवेत पानांच्या खालच्या बाजूस फिकट वाढ",
                "खोडावर किंवा फळांवर काळे डाग",
            ],
            "actions": [
                "लवकर कृती करा आणि जास्त बाधित भाग वेगळा करा.",
                "ओला बाधित भाग इतर झाडांवर नेऊ नका.",
                "रोग वेगाने पसरू शकतो; कृषी तज्ज्ञांचा त्वरित सल्ला घ्या.",
            ],
            "prevention": [
                "पानांवर पाणी पडणे टाळा.",
                "झाडांमध्ये योग्य अंतर ठेवा.",
                "टोमॅटो आणि बटाट्याच्या झाडांची नियमित तपासणी करा.",
            ],
        },
    },
    "Tomato___Leaf_Mold": {
        "en": {
            "name": "Leaf Mold",
            "kind": "Fungal",
            "urgency": "Medium",
            "cause": "Fungus",
            "summary": "Yellow patches on top of leaves may have olive-grey growth underneath.",
            "symptoms": [
                "Yellowish patches on the upper leaf surface",
                "Olive or grey growth underneath",
                "Leaves may curl or fall",
            ],
            "actions": [
                "Improve airflow around plants.",
                "Remove affected leaves carefully.",
                "Water at the base and reduce excess humidity.",
            ],
            "prevention": [
                "Avoid overcrowding.",
                "Water early so foliage can dry.",
                "Inspect both sides of leaves regularly.",
            ],
        },
        "mr": {
            "name": "पानांवरील बुरशी",
            "kind": "Fungal",
            "urgency": "Medium",
            "cause": "बुरशी",
            "summary": "पानांच्या वरच्या बाजूस पिवळे डाग आणि खाली राखाडी-जैतूनी वाढ दिसू शकते.",
            "symptoms": [
                "पानांच्या वरच्या बाजूस पिवळसर डाग",
                "खालच्या बाजूस राखाडी किंवा जैतूनी रंगाची वाढ",
                "पाने वाकडी होणे किंवा गळणे",
            ],
            "actions": [
                "झाडांमध्ये हवा खेळती ठेवा.",
                "बाधित पाने काळजीपूर्वक काढा.",
                "मुळाजवळ पाणी द्या आणि जास्त ओलावा कमी करा.",
            ],
            "prevention": [
                "झाडांची गर्दी टाळा.",
                "सकाळी पाणी द्या.",
                "पानांच्या दोन्ही बाजू नियमित तपासा.",
            ],
        },
    },
    "Tomato___Septoria_leaf_spot": {
        "en": {
            "name": "Septoria Leaf Spot",
            "kind": "Fungal",
            "urgency": "Medium",
            "cause": "Fungus",
            "summary": "Many small round spots with pale centres and dark edges.",
            "symptoms": [
                "Small circular spots with grey centres",
                "Dark borders around the spots",
                "Lower leaves may yellow and fall",
            ],
            "actions": [
                "Remove badly affected leaves.",
                "Clean tools after handling infected plants.",
                "Ask an expert if spotting spreads to newer leaves.",
            ],
            "prevention": [
                "Rotate crops.",
                "Use mulch to reduce soil splash.",
                "Avoid overhead watering.",
            ],
        },
        "mr": {
            "name": "सेप्टोरिया पानावरील ठिपके",
            "kind": "Fungal",
            "urgency": "Medium",
            "cause": "बुरशी",
            "summary": "पानांवर फिकट मध्यभाग आणि गडद कडा असलेले अनेक गोल डाग दिसतात.",
            "symptoms": [
                "राखाडी मध्यभाग असलेले लहान गोल डाग",
                "डागांभोवती गडद कडा",
                "खालची पाने पिवळी पडून गळू शकतात",
            ],
            "actions": [
                "जास्त बाधित पाने काढा.",
                "बाधित झाड हाताळल्यानंतर साधने स्वच्छ करा.",
                "नवीन पानांवर डाग पसरल्यास तज्ज्ञांचा सल्ला घ्या.",
            ],
            "prevention": [
                "पीक फेरपालट करा.",
                "माती उडू नये म्हणून आच्छादन वापरा.",
                "पानांवर पाणी शिंपडणे टाळा.",
            ],
        },
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "en": {
            "name": "Spider Mites",
            "kind": "Pest",
            "urgency": "Medium",
            "cause": "Tiny mites",
            "summary": "Tiny pale speckles and fine webbing can appear in hot, dry weather.",
            "symptoms": [
                "Fine pale speckles on leaves",
                "Leaves may turn bronze or dry",
                "Fine webbing under leaves",
            ],
            "actions": [
                "Inspect the undersides of leaves closely.",
                "Rinse leaves gently with water where appropriate.",
                "Ask an expert about suitable pest control if the problem persists.",
            ],
            "prevention": [
                "Avoid plant water stress.",
                "Check leaves weekly in hot weather.",
                "Protect beneficial insects where possible.",
            ],
        },
        "mr": {
            "name": "कोळी किडीचा प्रादुर्भाव",
            "kind": "Pest",
            "urgency": "Medium",
            "cause": "सूक्ष्म कोळी कीड",
            "summary": "उष्ण आणि कोरड्या हवेत पानांवर फिकट ठिपके व बारीक जाळी दिसू शकते.",
            "symptoms": [
                "पानांवर बारीक फिकट ठिपके",
                "पाने तांबूस होणे किंवा वाळणे",
                "पानांच्या खालच्या बाजूस बारीक जाळी",
            ],
            "actions": [
                "पानांच्या खालच्या बाजूची नीट तपासणी करा.",
                "योग्य असल्यास पानांवर हलक्या पाण्याने धुवा.",
                "प्रादुर्भाव कायम राहिल्यास योग्य नियंत्रणासाठी तज्ज्ञांचा सल्ला घ्या.",
            ],
            "prevention": [
                "झाडाला पाण्याचा ताण येऊ देऊ नका.",
                "उष्ण हवेत पानांची साप्ताहिक तपासणी करा.",
                "शक्य असल्यास उपयुक्त कीटकांचे संरक्षण करा.",
            ],
        },
    },
    "Tomato___Target_Spot": {
        "en": {
            "name": "Target Spot",
            "kind": "Fungal",
            "urgency": "Medium",
            "cause": "Fungus",
            "summary": "Brown ring-shaped spots may grow and merge on leaves.",
            "symptoms": [
                "Brown spots with ring patterns",
                "Spots may merge into larger patches",
                "Fruit may also develop dark marks",
            ],
            "actions": [
                "Remove affected leaves and plant debris.",
                "Improve airflow around the plant.",
                "Ask a local expert if the disease continues spreading.",
            ],
            "prevention": [
                "Space plants properly.",
                "Avoid overhead watering.",
                "Clean plant debris after harvest.",
            ],
        },
        "mr": {
            "name": "टार्गेट स्पॉट",
            "kind": "Fungal",
            "urgency": "Medium",
            "cause": "बुरशी",
            "summary": "पानांवर वलयांसारखे तपकिरी डाग दिसून ते एकत्र येऊ शकतात.",
            "symptoms": [
                "वलयांसारखे तपकिरी डाग",
                "डाग एकत्र येऊन मोठे होऊ शकतात",
                "फळांवरही काळसर डाग दिसू शकतात",
            ],
            "actions": [
                "बाधित पाने आणि झाडांचे अवशेष काढा.",
                "झाडाभोवती हवा खेळती ठेवा.",
                "रोग वाढत असल्यास स्थानिक तज्ज्ञांचा सल्ला घ्या.",
            ],
            "prevention": [
                "झाडांमध्ये योग्य अंतर ठेवा.",
                "पानांवर पाणी शिंपडणे टाळा.",
                "काढणीनंतर अवशेष स्वच्छ करा.",
            ],
        },
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "en": {
            "name": "Tomato Yellow Leaf Curl Virus",
            "kind": "Viral",
            "urgency": "High",
            "cause": "Virus commonly spread by whiteflies",
            "summary": "Leaves may curl upward and yellow, and plant growth may become stunted.",
            "symptoms": [
                "Upward curling and yellowing leaves",
                "Small or bushy-looking plants",
                "Flower drop and reduced fruit production",
            ],
            "actions": [
                "Separate severely affected plants and seek local advice.",
                "Inspect leaves for whiteflies.",
                "Ask an agriculture expert about suitable vector management.",
            ],
            "prevention": [
                "Use healthy seedlings and tolerant varieties where available.",
                "Protect seedlings with suitable insect netting.",
                "Control weeds that can host whiteflies.",
            ],
        },
        "mr": {
            "name": "टोमॅटो पिवळा पर्णकुंचन विषाणू",
            "kind": "Viral",
            "urgency": "High",
            "cause": "पांढरी माशी पसरवू शकणारा विषाणू",
            "summary": "पाने वरच्या बाजूला वाकून पिवळी पडू शकतात आणि झाडाची वाढ खुंटू शकते.",
            "symptoms": [
                "पाने वरच्या बाजूला वाकणे व पिवळी पडणे",
                "झाडाची वाढ खुंटणे",
                "फुले गळणे आणि फळधारणा कमी होणे",
            ],
            "actions": [
                "जास्त बाधित झाडांबाबत स्थानिक कृषी तज्ज्ञांचा सल्ला घ्या.",
                "पानांवर पांढऱ्या माशीची तपासणी करा.",
                "कीड नियंत्रणासाठी योग्य मार्गदर्शन घ्या.",
            ],
            "prevention": [
                "निरोगी रोपे आणि उपलब्ध असल्यास सहनशील वाण वापरा.",
                "रोपांचे योग्य जाळीने संरक्षण करा.",
                "पांढऱ्या माशीला आश्रय देणारी तणे नियंत्रित करा.",
            ],
        },
    },
    "Tomato___Tomato_mosaic_virus": {
        "en": {
            "name": "Tomato Mosaic Virus",
            "kind": "Viral",
            "urgency": "Medium",
            "cause": "Virus spread through contaminated hands, tools or plant material",
            "summary": "Leaves may show irregular light and dark green patches.",
            "symptoms": [
                "Mottled light and dark green areas",
                "Narrow, curled or distorted leaves",
                "Uneven fruit appearance in some cases",
            ],
            "actions": [
                "Seek expert advice about affected plants.",
                "Wash hands and clean tools after handling plants.",
                "Avoid touching healthy plants immediately after infected ones.",
            ],
            "prevention": [
                "Use healthy seeds and seedlings.",
                "Clean tools and stakes between uses.",
                "Remove infected plant debris carefully.",
            ],
        },
        "mr": {
            "name": "टोमॅटो मोझॅक विषाणू",
            "kind": "Viral",
            "urgency": "Medium",
            "cause": "हात, साधने किंवा बाधित वनस्पतींच्या भागातून पसरू शकणारा विषाणू",
            "summary": "पानांवर फिकट आणि गडद हिरव्या रंगाचे अनियमित पट्टे दिसू शकतात.",
            "symptoms": [
                "फिकट व गडद हिरवे ठिपके किंवा पट्टे",
                "पाने अरुंद, वाकडी किंवा विकृत होणे",
                "काही वेळा फळांचा रंग असमान होणे",
            ],
            "actions": [
                "बाधित झाडांसाठी तज्ज्ञांचा सल्ला घ्या.",
                "झाड हाताळल्यानंतर हात धुवा आणि साधने स्वच्छ करा.",
                "बाधित झाडानंतर लगेच निरोगी झाडाला हात लावू नका.",
            ],
            "prevention": [
                "निरोगी बियाणे आणि रोपे वापरा.",
                "साधने आणि आधारकाठ्या स्वच्छ करा.",
                "बाधित वनस्पतींचे अवशेष काळजीपूर्वक काढा.",
            ],
        },
    },
    "Tomato___healthy": {
        "en": {
            "name": "Healthy Leaf",
            "kind": "Healthy",
            "urgency": "Low",
            "cause": "No disease class selected by the model",
            "summary": "The model considers this leaf most similar to healthy tomato leaves.",
            "symptoms": [
                "Generally even green colour",
                "No obvious disease spots in the photo",
                "Normal leaf shape may be visible",
            ],
            "actions": [
                "Continue regular watering at the base.",
                "Inspect both sides of leaves weekly.",
                "Monitor new growth for changes.",
            ],
            "prevention": [
                "Give plants enough space and airflow.",
                "Use clean tools and healthy seedlings.",
                "Remove fallen leaves and plant debris.",
            ],
        },
        "mr": {
            "name": "निरोगी पान",
            "kind": "Healthy",
            "urgency": "Low",
            "cause": "मॉडेलनुसार रोगाची स्पष्ट लक्षणे दिसत नाहीत",
            "summary": "मॉडेलनुसार हे पान निरोगी टोमॅटोच्या पानांसारखे दिसते.",
            "symptoms": [
                "साधारणपणे एकसारखा हिरवा रंग",
                "फोटोमध्ये स्पष्ट रोगाचे डाग दिसत नाहीत",
                "पानांचा आकार सामान्य असू शकतो",
            ],
            "actions": [
                "झाडाच्या मुळाजवळ नियमित पाणी द्या.",
                "आठवड्यातून एकदा पानांच्या दोन्ही बाजू तपासा.",
                "नवीन पानांमध्ये बदल होतो का ते पाहा.",
            ],
            "prevention": [
                "झाडांमध्ये योग्य अंतर आणि हवा खेळती ठेवा.",
                "स्वच्छ साधने आणि निरोगी रोपे वापरा.",
                "गळलेली पाने आणि अवशेष काढा.",
            ],
        },
    },
}

# =========================================================
# PASTEL UI
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

:root {
    --paper: #F8F7F0;
    --ink: #405448;
    --sage: #739780;
    --sage-light: #E4F1E7;
    --blue: #DDEBF5;
    --peach: #FCE8CF;
    --lavender: #F2E5F2;
    --white: #FFFFFF;
    --line: #E3E7DD;
}
.stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
h1, h2, h3, h4 {
    color: var(--ink) !important;
    font-family: 'Playfair Display', Georgia, serif !important;
}
p, li, label, span {
    color: var(--ink);
}
[data-testid="stSidebar"] {
    background: var(--sage-light);
}
div.stButton > button {
    border-radius: 14px;
    border: 1px solid #D6E2D7;
    background: white;
    color: #405448;
    font-weight: 600;
    min-height: 42px;
}
div.stButton > button:hover {
    background: #E4F1E7;
    border-color: #739780;
    color: #405448;
}
div.stButton > button[kind="primary"] {
    background: #739780;
    color: white;
    border-color: #739780;
}
div[data-testid="stFileUploader"] {
    background: white;
    border-radius: 16px;
    padding: 10px;
}
.ag-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 22px;
    margin: 8px 0 14px 0;
    box-shadow: 0 4px 16px rgba(64,84,72,.04);
}
.ag-hero {
    background: linear-gradient(135deg, #E4F1E7, #DDEBF5);
    border-radius: 26px;
    padding: 32px;
    margin: 12px 0 22px;
}
.ag-pill {
    display: inline-block;
    background: white;
    border: 1px solid #DCE7DC;
    border-radius: 50px;
    padding: 6px 12px;
    font-size: .85rem;
    margin-bottom: 8px;
}
.ag-small {
    font-size: .9rem;
    opacity: .85;
}
.ag-bar {
    height: 9px;
    background: #E9EDE6;
    border-radius: 10px;
    overflow: hidden;
}
.ag-bar-inner {
    height: 9px;
    background: #739780;
    border-radius: 10px;
}
.disease-card {
    background: #FFFFFF; border: 1px solid #E3E7DD; border-radius: 20px;
    padding: 14px; margin: 6px 0 16px; box-shadow: 0 5px 18px rgba(64,84,72,.06);
    min-height: 100%;
}
.disease-card h3 { margin: 8px 0 5px; font-size: 1.12rem; }
.disease-tag { display:inline-block; border-radius:999px; padding:4px 10px;
    background:#E4F1E7; color:#405448; font-size:.78rem; font-weight:600; }
.care-intro { background:linear-gradient(135deg,#E4F1E7,#F2E5F2);
    border-radius:22px; padding:22px 24px; margin:8px 0 20px; }
.care-panel { background:#FFFFFF; border:1px solid #E3E7DD; border-radius:18px;
    padding:18px 20px; margin:0 0 14px; box-shadow:0 4px 14px rgba(64,84,72,.04); }
.care-panel h3 { margin-top:0; margin-bottom:10px; }
.care-panel li { margin-bottom:8px; }
.ag-footer {
    margin-top: 36px;
    border-top: 1px solid var(--line);
    padding-top: 18px;
    font-size: .85rem;
    opacity: .85;
}
hr { border-color: #E3E7DD !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LANGUAGE
# =========================================================
LANG = {
    "English": {
        "home": "Home",
        "check": "Check a Leaf",
        "guide": "Disease Library",
        "history": "History",
        "care": "Care Guide",
        "about": "About",
        "upload": "Upload a tomato leaf photo",
        "camera": "Or take a photo",
        "analyze": "Analyze leaf",
        "result": "Diagnosis result",
        "symptoms": "Symptoms",
        "actions": "What to do",
        "prevention": "Prevention",
        "confidence": "Model confidence",
        "download": "Download PDF report",
        "damage": "Experimental leaf-damage estimate",
        "history_empty": "No diagnosis records yet. Analyze a leaf to create your first record.",
        "clear": "Clear history",
        "saved": "Diagnosis saved to history.",
        "choose": "Choose language",
        "not_leaf": "The model is uncertain. Use a clear close-up photo of a single tomato leaf.",
        "disclaimer": "AI predictions can be wrong. Confirm the result with an agriculture expert before treatment.",
    },
    "मराठी": {
        "home": "मुख्यपृष्ठ",
        "check": "पान तपासा",
        "guide": "रोग माहिती",
        "history": "इतिहास",
        "care": "पीक काळजी",
        "about": "माहिती",
        "upload": "टोमॅटोच्या पानाचा फोटो अपलोड करा",
        "camera": "किंवा फोटो काढा",
        "analyze": "पान तपासा",
        "result": "तपासणीचा निकाल",
        "symptoms": "लक्षणे",
        "actions": "काय करावे",
        "prevention": "प्रतिबंध",
        "confidence": "मॉडेलचा विश्वास",
        "download": "PDF अहवाल डाउनलोड करा",
        "damage": "पानावरील नुकसानाचा प्रायोगिक अंदाज",
        "history_empty": "अजून कोणतीही नोंद नाही. पान तपासल्यावर नोंद येथे दिसेल.",
        "clear": "इतिहास हटवा",
        "saved": "तपासणी इतिहासात जतन केली.",
        "choose": "भाषा निवडा",
        "not_leaf": "मॉडेलचा अंदाज निश्चित नाही. एका टोमॅटोच्या पानाचा स्पष्ट जवळचा फोटो वापरा.",
        "disclaimer": "AI चा अंदाज चुकीचा असू शकतो. उपचार करण्यापूर्वी कृषी तज्ज्ञांचा सल्ला घ्या.",
    },
}

# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        result = gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        if not result or not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Could not download the model. Check that the Google Drive file "
                "is accessible to anyone with the link."
            )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Model output count is {model.output_shape[-1]}, "
            f"but {len(CLASS_NAMES)} labels are configured."
        )
    return model


def prepare_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    # Keep raw 0–255 pixel values to match the original training pipeline.
    return np.expand_dims(arr, axis=0)


def predict_disease(model, image):
    predictions = model.predict(prepare_image(image), verbose=0)[0]
    tomato_probs = predictions[TOMATO_INDICES]
    tomato_share = float(tomato_probs.sum())

    if not np.isfinite(tomato_share) or tomato_share <= 0:
        raise ValueError("The model returned invalid prediction scores.")

    # This normalizes the scores only within tomato classes.
    normalized = tomato_probs / tomato_share
    order = np.argsort(normalized)[::-1]
    ranked = [
        (CLASS_NAMES[TOMATO_INDICES[i]], float(normalized[i]))
        for i in order
    ]
    return ranked, tomato_share


# =========================================================
# HISTORY
# =========================================================
def load_history():
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_history(records):
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as file:
            json.dump(records[-100:], file, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


if "history" not in st.session_state:
    st.session_state.history = load_history()
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "language" not in st.session_state:
    st.session_state.language = "English"


# =========================================================
# PDF REPORT
# =========================================================
def create_pdf_report(record, language):
    """Create a readable English PDF, including when the app is in Marathi.

    ReportLab's default Helvetica fonts do not include Devanagari glyphs. To avoid
    the black-square issue, the PDF consistently uses English labels and content.
    The web app itself remains bilingual.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    except ImportError:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=44, leftMargin=44, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="AgriTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=25, leading=30, textColor=colors.HexColor("#405448"), alignment=TA_CENTER, spaceAfter=5))
    styles.add(ParagraphStyle(name="AgriSubtitle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=11, leading=15, textColor=colors.HexColor("#6B7E70"), alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name="AgriHeading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, leading=17, textColor=colors.HexColor("#547C61"), spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="AgriBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.5, leading=14, textColor=colors.HexColor("#405448"), spaceAfter=5))
    styles.add(ParagraphStyle(name="AgriNote", parent=styles["Italic"], fontName="Helvetica-Oblique",
        fontSize=8.5, leading=12, textColor=colors.HexColor("#6B756C"), spaceBefore=16))

    # Always take report copy from the English disease library to avoid unsupported Marathi glyphs.
    class_name = record.get("class_name", "Tomato___healthy")
    english_info = DISEASES.get(class_name, {}).get("en", {})
    display_name = english_info.get("name", record.get("display_name", "Tomato leaf condition"))
    kind = english_info.get("kind", "Tomato")
    symptoms = english_info.get("symptoms", record.get("symptoms", []))
    actions = english_info.get("actions", record.get("actions", []))
    prevention = english_info.get("prevention", record.get("prevention", []))

    story = [Paragraph("AgriShield", styles["AgriTitle"]),
             Paragraph("TOMATO LEAF ANALYSIS REPORT", styles["AgriSubtitle"])]
    meta = [
        [Paragraph("<b>Date and time</b>", styles["AgriBody"]), Paragraph(str(record.get("date", "Not recorded")), styles["AgriBody"])],
        [Paragraph("<b>Predicted condition</b>", styles["AgriBody"]), Paragraph(display_name, styles["AgriBody"])],
        [Paragraph("<b>Category</b>", styles["AgriBody"]), Paragraph(str(kind), styles["AgriBody"])],
        [Paragraph("<b>Model confidence</b>", styles["AgriBody"]), Paragraph(f"{float(record.get('confidence', 0)):.1f}%", styles["AgriBody"])],
    ]
    if record.get("damage") is not None:
        meta.append([Paragraph("<b>Experimental leaf-damage estimate</b>", styles["AgriBody"]),
                     Paragraph(f"{record.get('damage')}% (rough visual estimate)", styles["AgriBody"])])
    table = Table(meta, colWidths=[165, 300], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F7F1")),
        ("BOX", (0, 0), (-1, -1), .6, colors.HexColor("#DCE7DC")),
        ("INNERGRID", (0, 0), (-1, -1), .35, colors.HexColor("#E3E7DD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([table, Spacer(1, 10)])

    for title, items in [("Symptoms to look for", symptoms),
                         ("Recommended next steps", actions),
                         ("Prevention tips", prevention)]:
        section = [Paragraph(title, styles["AgriHeading"])]
        for item in items:
            safe_item = str(item).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            section.append(Paragraph("&#8226; &nbsp;" + safe_item, styles["AgriBody"]))
        story.append(KeepTogether(section))

    story.append(Paragraph(
        "Important: This is an AI-generated educational estimate, not a confirmed agricultural diagnosis. "
        "Model confidence is not the same as real-world accuracy. Consult a local agriculture expert before treatment.",
        styles["AgriNote"]))
    doc.build(story)
    return buffer.getvalue()


# =========================================================
# EXPERIMENTAL LEAF DAMAGE ESTIMATE
# =========================================================
def estimate_leaf_damage(image):
    """
    Rough green-pixel estimate. This is experimental and not validated.
    It cannot reliably separate a leaf from its background.
    """
    arr = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    green_mask = (g > r * 1.08) & (g > b * 1.02) & (g > 35)
    total = green_mask.size
    green_pixels = int(green_mask.sum())

    if green_pixels < 100:
        return None

    # Only counts pixels that do not meet this simple green rule.
    # Background pixels may inflate this estimate.
    non_green = total - green_pixels
    return round((non_green / total) * 100, 1)


# =========================================================
# DISPLAY HELPERS
# =========================================================
def get_info(class_name, language):
    lang_key = "mr" if language == "मराठी" else "en"
    return DISEASES[class_name][lang_key]


def show_bullets(items):
    for item in items:
        st.markdown(f"- {item}")


def show_diagnosis(record, language):
    info = get_info(record["class_name"], language)
    t = LANG[language]

    st.markdown(f"### {t['result']}")
    st.markdown(
        f'<div class="ag-card"><span class="ag-pill">{info["kind"]}</span>'
        f'<h2>{info["name"]}</h2><p>{info["summary"]}</p>'
        f'<p><b>{t["confidence"]}:</b> {record["confidence"]:.1f}%</p>'
        f'<div class="ag-bar"><div class="ag-bar-inner" '
        f'style="width:{min(100, max(0, record["confidence"])):.1f}%"></div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if record["confidence"] < 60 or record.get("tomato_share", 1) < 0.5:
        st.warning(t["not_leaf"])

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"#### {t['symptoms']}")
        show_bullets(info["symptoms"])
    with right:
        st.markdown(f"#### {t['actions']}")
        show_bullets(info["actions"])

    st.markdown(f"#### {t['prevention']}")
    show_bullets(info["prevention"])

    st.info(t["disclaimer"])

    if record.get("damage") is not None:
        st.markdown(f"#### {t['damage']}")
        st.metric("Estimated non-green pixels", f'{record["damage"]}%')
        st.caption(
            "Experimental visual estimate only. It may count background pixels "
            "as damage and must not be used to decide treatment."
        )

    pdf = create_pdf_report(record, language)
    if pdf:
        st.download_button(
            t["download"],
            data=pdf,
            file_name="AgriShield_Diagnosis_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning(
            "PDF reports require ReportLab. Add reportlab to requirements.txt "
            "and redeploy the app."
        )


# =========================================================
# TOP NAVIGATION
# =========================================================
language_col, title_col = st.columns([1, 3], vertical_alignment="center")
with language_col:
    st.session_state.language = st.selectbox(
        "Language / भाषा",
        ["English", "मराठी"],
        index=0 if st.session_state.language == "English" else 1,
    )

language = st.session_state.language
t = LANG[language]

with title_col:
    st.markdown(
        "<h1 style='text-align:right;margin-bottom:0'>🍅 AgriShield</h1>"
        "<p style='text-align:right;margin-top:0'>A little care goes a long way.</p>",
        unsafe_allow_html=True,
    )

pages = [
    ("Home", t["home"]),
    ("Check", t["check"]),
    ("Library", t["guide"]),
    ("History", t["history"]),
    ("Care", t["care"]),
    ("About", t["about"]),
]
nav_cols = st.columns(len(pages))
for col, (page_key, label) in zip(nav_cols, pages):
    with col:
        if st.button(
            label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if st.session_state.page == page_key else "secondary",
        ):
            st.session_state.page = page_key
            st.rerun()

st.divider()

# =========================================================
# HOME
# =========================================================
if st.session_state.page == "Home":
    if language == "मराठी":
        hero_tag = "वनस्पतींचे आरोग्य · सोप्या पद्धतीने"
        hero_title = "निरोगी टोमॅटोच्या झाडाची सुरुवात एका पानापासून."
        hero_copy = "टोमॅटोच्या पानाचा फोटो तपासा, संभाव्य रोग समजून घ्या आणि पुढील काळजी जाणून घ्या."
        steps = [("०१ · तपासा", "स्पष्ट फोटो अपलोड करा किंवा कॅमेरा वापरा."),
                 ("०२ · समजून घ्या", "संभाव्य निकाल आणि मॉडेलचा विश्वास पाहा."),
                 ("०३ · काळजी घ्या", "काळजी आणि प्रतिबंधाच्या सूचना वाचा.")]
        what_title = "येथे तुम्ही काय करू शकता"
        what_copy = "टोमॅटो रोग माहिती पाहा, अलीकडील तपासण्या पहा, मराठी किंवा इंग्रजी निवडा आणि PDF अहवाल डाउनलोड करा."
        warning = "AgriShield हा विद्यार्थी प्रकल्प आहे. AI चा अंदाज चुकीचा असू शकतो; उपचारापूर्वी कृषी तज्ज्ञांचा सल्ला घ्या."
    else:
        hero_tag = "PLANT HEALTH · MADE SIMPLE"
        hero_title = "A healthier tomato plant starts with one leaf."
        hero_copy = "Check a tomato leaf, learn about possible diseases, and discover practical next steps in one friendly place."
        steps = [("01 · Check", "Upload a clear photo or use your camera."),
                 ("02 · Understand", "Read the likely result and its confidence."),
                 ("03 · Take action", "Review care and prevention suggestions.")]
        what_title = "What you can do here"
        what_copy = "Explore a tomato disease library, view your recent checks, switch between English and Marathi, and download a PDF report after a diagnosis."
        warning = "AgriShield is a student project. AI predictions can be wrong, so confirm the result with an agriculture expert before treating a plant."

    st.markdown(f"""<div class="ag-hero"><span class="ag-pill">{hero_tag}</span>
        <h1 style="font-size:2.45rem;line-height:1.15">{hero_title}</h1>
        <p style="font-size:1.08rem;max-width:760px">{hero_copy}</p></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    for col, (heading, copy), bg in zip([c1,c2,c3], steps, ["#E4F1E7", "#DDEBF5", "#FCE8CF"]):
        with col:
            st.markdown(f'<div class="ag-card" style="background:{bg};min-height:125px"><h3>{heading}</h3><p>{copy}</p></div>', unsafe_allow_html=True)
    st.markdown(f"### {what_title}")
    st.write(what_copy)
    st.warning(warning)

# =========================================================
# CHECK A LEAF
# =========================================================
elif st.session_state.page == "Check":
    st.title(t["check"])
    st.write(
        "For the best result, photograph one tomato leaf in good light, close up, "
        "with as little background as possible."
    )

    upload_col, camera_col = st.columns(2, gap="large")
    with upload_col:
        uploaded = st.file_uploader(
            t["upload"],
            type=["jpg", "jpeg", "png"],
            key="leaf_upload",
        )
    with camera_col:
        camera = st.camera_input(t["camera"], key="leaf_camera")

    source = camera if camera is not None else uploaded

    if source is not None:
        try:
            image = Image.open(source).convert("RGB")
            st.image(image, caption="Selected leaf photo", use_container_width=True)

            with st.expander("Photo tips"):
                st.markdown(
                    "- Use a clear, focused image.\n"
                    "- Avoid shadows and very dark photos.\n"
                    "- Photograph one leaf rather than the whole plant.\n"
                    "- If possible, inspect both sides of the leaf."
                )

            if st.button(t["analyze"], type="primary", use_container_width=True):
                with st.spinner("Loading the model and checking the leaf..."):
                    model = load_model()
                    ranked, tomato_share = predict_disease(model, image)
                    best_class, confidence = ranked[0]
                    info = get_info(best_class, language)

                    damage = estimate_leaf_damage(image)
                    record = {
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "class_name": best_class,
                        "display_name": info["name"],
                        "confidence": round(confidence * 100, 1),
                        "tomato_share": round(tomato_share, 4),
                        "damage": damage,
                        "report_language": language,
                        "symptoms": info["symptoms"],
                        "actions": info["actions"],
                        "prevention": info["prevention"],
                    }

                    st.session_state.latest_result = record
                    st.session_state.history.insert(0, record)
                    st.session_state.history = st.session_state.history[:100]
                    saved = save_history(st.session_state.history)

                if saved:
                    st.success(t["saved"])
                else:
                    st.warning(
                        "The result is available in this session, but the history "
                        "could not be saved to the local file."
                    )

        except Exception as error:
            st.error(
                "The image could not be analyzed. Check the model file and its "
                "Google Drive sharing settings."
            )
            st.exception(error)

    if st.session_state.latest_result is not None:
        st.divider()
        show_diagnosis(st.session_state.latest_result, language)

# =========================================================
# DISEASE LIBRARY
# =========================================================
elif st.session_state.page == "Library":
    st.title(t["guide"])
    if language == "मराठी":
        st.markdown('<div class="care-intro"><h3>टोमॅटो पानांची छोटी मार्गदर्शिका</h3><p>लक्षणे, काळजी आणि प्रतिबंध जाणून घेण्यासाठी कोणत्याही कार्डवरील माहिती उघडा.</p></div>', unsafe_allow_html=True)
        filter_label = "प्रकारानुसार निवडा"
        kind_options = ["सर्व", "Fungal", "Fungal-like", "Bacterial", "Viral", "Pest", "Healthy"]
        kind_map = {"सर्व":"All", "बुरशीजन्य":"Fungal"}
    else:
        st.markdown('<div class="care-intro"><h3>Know the leaf. Care with confidence.</h3><p>Explore common tomato conditions, recognise possible symptoms, and learn practical care steps.</p></div>', unsafe_allow_html=True)
        filter_label = "Filter by category"
        kind_options = ["All", "Fungal", "Fungal-like", "Bacterial", "Viral", "Pest", "Healthy"]
    chosen_kind = st.selectbox(filter_label, kind_options, key="disease_filter")
    chosen_kind_en = "All" if chosen_kind in ("All", "सर्व") else chosen_kind

    visible = []
    for class_name in DISEASES:
        info = get_info(class_name, language)
        if chosen_kind_en != "All" and info["kind"] != chosen_kind_en:
            continue
        visible.append((class_name, info))

    # Two-column image cards, with details tucked into a styled expander.
    for offset in range(0, len(visible), 2):
        cols = st.columns(2, gap="large")
        for col, (class_name, info) in zip(cols, visible[offset:offset+2]):
            with col:
                image_path = DISEASE_IMAGES.get(class_name)
                if image_path and os.path.exists(image_path):
                    st.image(image_path, use_container_width=True)
                st.markdown(f'<div class="disease-card"><span class="disease-tag">{info["kind"]}</span><h3>{info["name"]}</h3><p>{info["summary"]}</p></div>', unsafe_allow_html=True)
                with st.expander("माहिती आणि काळजी पहा" if language == "मराठी" else "Explore symptoms & care"):
                    st.markdown(f"**{t['symptoms']}**")
                    show_bullets(info["symptoms"])
                    st.markdown(f"**{t['actions']}**")
                    show_bullets(info["actions"])
                    st.markdown(f"**{t['prevention']}**")
                    show_bullets(info["prevention"])

# =========================================================
# HISTORY
# =========================================================
elif st.session_state.page == "History":
    st.title(t["history"])
    records = st.session_state.history

    if not records:
        st.info(t["history_empty"])
    else:
        st.caption(
            "History is stored in a local JSON file. On Streamlit Cloud, it may "
            "not survive app restarts and should not be treated as a private "
            "multi-user database."
        )

        if st.button(t["clear"], type="secondary"):
            st.session_state.history = []
            st.session_state.latest_result = None
            saved = save_history([])
            if saved:
                st.success("History cleared.")
            else:
                st.warning("History cleared for this session, but the file could not be updated.")
            st.rerun()

        for i, record in enumerate(records):
            with st.expander(
                f'{record.get("date", "Unknown date")} · '
                f'{record.get("display_name", "Unknown result")} · '
                f'{record.get("confidence", 0):.1f}%'
            ):
                st.write(f'**{t["confidence"]}:** {record.get("confidence", 0):.1f}%')
                st.write(f'**{t["symptoms"]}:**')
                show_bullets(record.get("symptoms", []))
                st.write(f'**{t["actions"]}:**')
                show_bullets(record.get("actions", []))

                pdf = create_pdf_report(record, language)
                if pdf:
                    st.download_button(
                        "Download this report",
                        data=pdf,
                        file_name=f"AgriShield_Report_{i + 1}.pdf",
                        mime="application/pdf",
                        key=f"history_pdf_{i}",
                    )

# =========================================================
# CARE GUIDE
# =========================================================
elif st.session_state.page == "Care":
    st.title(t["care"])
    if language == "मराठी":
        st.markdown('<div class="care-intro"><h3>लहान सवयी, निरोगी झाडे 🌿</h3><p>टोमॅटोच्या झाडांची नियमित काळजी घेतल्याने काही सामान्य समस्यांचा धोका कमी होऊ शकतो.</p></div>', unsafe_allow_html=True)
        care_sections = [
            ("☀️ सूर्यप्रकाश आणि अंतर", ["टोमॅटोच्या झाडांना शक्यतो दररोज ६–८ तास सूर्यप्रकाश मिळू द्या.", "झाडांमध्ये हवा खेळती राहील असे अंतर ठेवा."]),
            ("💧 पाणी देणे", ["झाडाच्या मुळाजवळ पाणी द्या.", "मातीचा ओलावा तपासून पाणी द्या; पाणी साचू देऊ नका.", "पानांवर सतत पाणी राहणे टाळा."]),
            ("🍃 स्वच्छता", ["गळलेली आणि बाधित पाने काढा.", "बाधित झाड हाताळल्यानंतर साधने स्वच्छ करा.", "बाधित अवशेषांची योग्य विल्हेवाट लावा."]),
            ("🔎 नियमित तपासणी", ["पानांच्या वरच्या आणि खालच्या बाजू तपासा.", "पानांवरील डाग, पिवळेपणा किंवा बारीक जाळी पाहा.", "रोग पसरत असल्यास कृषी तज्ज्ञांचा सल्ला घ्या."]),
        ]
    else:
        st.markdown('<div class="care-intro"><h3>Small habits. Healthier plants. 🌿</h3><p>A simple routine can help you notice changes early and keep tomato plants in better growing conditions.</p></div>', unsafe_allow_html=True)
        care_sections = [
            ("☀️ Sunlight and spacing", ["Aim for around 6–8 hours of sunlight when practical.", "Space plants so air can move between them."]),
            ("💧 Watering", ["Water near the base of the plant.", "Check soil moisture and avoid waterlogging.", "Avoid keeping leaves wet for long periods."]),
            ("🍃 Cleanliness", ["Remove fallen and badly affected leaves.", "Clean tools after handling affected plants.", "Dispose of diseased debris appropriately."]),
            ("🔎 Regular inspection", ["Check the upper and lower sides of leaves.", "Look for spots, yellowing or fine webbing.", "Seek expert advice if symptoms spread."]),
        ]
    cols = st.columns(2, gap="large")
    for index, (title, points) in enumerate(care_sections):
        with cols[index % 2]:
            st.markdown(f'<div class="care-panel"><h3>{title}</h3><ul>' + ''.join(f'<li>{point}</li>' for point in points) + '</ul></div>', unsafe_allow_html=True)
    st.info("ही सर्वसाधारण काळजीची माहिती आहे; रोगाचे निदान किंवा उपचारांसाठी कृषी तज्ज्ञांचा सल्ला घ्या." if language == "मराठी" else "These are general care tips, not a diagnosis or treatment plan. Ask a local agriculture expert for disease-specific advice.")

# =========================================================
# ABOUT
# =========================================================
elif st.session_state.page == "About":
    st.title(t["about"])
    if language == "मराठी":
        st.write(
            "AgriShield हा टोमॅटोच्या पानांवरील संभाव्य रोग ओळखण्यासाठी तयार केलेला "
            "विद्यार्थी प्रकल्प आहे. हा CNN आधारित मॉडेल वापरतो. निकाल हा केवळ "
            "प्राथमिक अंदाज आहे; तो तज्ज्ञांच्या निदानाची जागा घेत नाही."
        )
        st.markdown("### वापरलेली तंत्रज्ञाने")
        show_bullets([
            "Python",
            "Streamlit — वेबसाइट इंटरफेस",
            "TensorFlow आणि Keras — प्रशिक्षित मॉडेल",
            "NumPy आणि Pillow — प्रतिमा प्रक्रिया",
            "JSON — स्थानिक इतिहास नोंद",
        ])
    else:
        st.write(
            "AgriShield is a student project that uses a trained CNN model to "
            "estimate which of nine tomato conditions, or a healthy leaf, best "
            "matches an uploaded photo. It is intended as an educational first check."
        )
        st.markdown("### Technologies used")
        show_bullets([
            "Python",
            "Streamlit for the web interface",
            "TensorFlow and Keras for model inference",
            "NumPy and Pillow for image processing",
            "JSON for local history storage",
            "ReportLab for PDF reports",
        ])

    st.warning(
        "Model confidence is not the same as real-world accuracy. The damage "
        "estimate is experimental, may include background pixels, and should "
        "not be used as a treatment decision."
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="ag-footer">🍃 <b>AgriShield</b> · Student project · '
    'AI predictions should be confirmed by an agriculture expert.</div>',
    unsafe_allow_html=True,
)
