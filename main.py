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
