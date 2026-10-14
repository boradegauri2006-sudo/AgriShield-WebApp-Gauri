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
    page_title="AgriShield | Tomato leaf disease check",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed"
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

# --------------------------------------------------
# DISEASE CONTENT
# --------------------------------------------------
KIND_COLORS = {
    "Fungal": ("#8A5A12", "#F6E8CC"),
    "Bacterial": ("#9C2D20", "#F8DDD8"),
    "Viral": ("#5B3F8C", "#E7DFF4"),
    "Pest": ("#1B6B6B", "#D6EEEE"),
    "Healthy": ("#1D5C2E", "#DCEFD9"),
}

URGENCY_COLORS = {
    "Low": "#2E6A3E",
    "Medium": "#C98A1B",
    "High": "#D23B2A",
}

TOMATO_DISEASES = {
    "Tomato___Bacterial_spot": {
        "name": "Bacterial spot",
        "kind": "Bacterial",
        "urgency": "Medium",
        "cause": "Bacteria (Xanthomonas species)",
        "summary": "Small, dark, water-soaked spots on leaves and rough, scabby spots on fruit.",
        "symptoms": [
            "Tiny water-soaked spots on leaves that turn dark brown or black",
            "Spots may have a yellow halo and can make leaves look ragged",
            "Raised, rough, scab-like spots on green fruit",
        ],
        "conditions": "Warm, wet weather. Splashing rain or overhead watering spreads the bacteria. It can also arrive on infected seed or seedlings.",
        "act": [
            "Remove badly spotted leaves and keep them away from the plants",
            "Stop overhead watering and avoid working in the plants while they are wet",
            "Ask your local agriculture office whether a copper-based spray is suitable for your area",
        ],
        "prevent": [
            "Buy certified disease-free seed and healthy seedlings",
            "Rotate tomatoes with non-host crops for 2 to 3 years",
            "Water at the base and mulch to stop soil splash",
        ],
    },
    "Tomato___Early_blight": {
        "name": "Early blight",
        "kind": "Fungal",
        "urgency": "Medium",
        "cause": "Fungus (Alternaria solani)",
        "summary": "Brown spots with ring patterns, usually starting on the oldest, lowest leaves.",
        "symptoms": [
            "Brown spots with concentric rings, like a target",
            "Yellowing around the spots, then the whole leaf dies",
            "Dark, sunken lesions near the stem or on fruit shoulders",
        ],
        "conditions": "Warm temperatures with alternating wet and dry spells. The fungus survives on old plant debris and in the soil.",
        "act": [
            "Pick off affected lower leaves and dispose of them, not in the compost",
            "Keep leaves dry and give plants room for airflow",
            "If it keeps spreading, ask an agriculture expert about a suitable fungicide",
        ],
        "prevent": [
            "Rotate crops and clear plant debris at the end of the season",
            "Mulch to keep soil off the lower leaves",
            "Stake plants and prune lower leaves so air moves freely",
        ],
    },
    "Tomato___Late_blight": {
        "name": "Late blight",
        "kind": "Fungal",
        "urgency": "High",
        "cause": "Water mold (Phytophthora infestans)",
        "summary": "Large, greasy grey-green patches that spread fast in cool, wet weather.",
        "symptoms": [
            "Large, irregular, water-soaked patches on leaves that turn brown",
            "White fuzzy growth under the leaves in humid conditions",
            "Dark, firm, greasy-looking patches on stems and fruit",
        ],
        "conditions": "Cool nights, high humidity and rain. Spores travel on wind and can wipe out a crop within days. Potatoes get the same disease.",
        "act": [
            "Act quickly: remove and bag infected plants, and do not compost them",
            "Check nearby tomato and potato plants every day",
            "Contact your local agriculture office early because this disease spreads fast",
        ],
        "prevent": [
            "Plant resistant varieties where available",
            "Avoid wetting leaves and give plants wide spacing",
            "Do not grow tomatoes next to potatoes",
        ],
    },
    "Tomato___Leaf_Mold": {
        "name": "Leaf mold",
        "kind": "Fungal",
        "urgency": "Low",
        "cause": "Fungus (Passalora fulva)",
        "summary": "Pale yellow patches on top of the leaf with olive-grey mold underneath.",
        "symptoms": [
            "Pale green or yellow spots on the upper leaf surface",
            "Velvety olive-green to grey mold on the underside of the same spot",
            "Leaves curl, wither and drop in severe cases",
        ],
        "conditions": "Very humid, still air, common in greenhouses and polytunnels.",
        "act": [
            "Improve ventilation and open vents or space the plants out",
            "Remove affected leaves and water at the base only",
            "Water early in the day so leaves dry quickly",
        ],
        "prevent": [
            "Keep humidity below roughly 85 percent in covered growing spaces",
            "Prune to open up the canopy",
            "Choose resistant varieties for greenhouse growing",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Septoria leaf spot",
        "kind": "Fungal",
        "urgency": "Medium",
        "cause": "Fungus (Septoria lycopersici)",
        "summary": "Many small round spots with grey centers and dark edges on lower leaves.",
        "symptoms": [
            "Numerous small circular spots with grey-white centers and dark brown borders",
            "Tiny black dots in the middle of the spots",
            "Lower leaves turn yellow and drop, exposing fruit to sun",
        ],
        "conditions": "Warm, wet, humid weather. Rain splash carries spores from soil and debris up to the leaves.",
        "act": [
            "Remove the worst leaves and keep foliage dry",
            "Clean tools between plants",
            "Ask an agriculture expert about fungicide if the spots reach the upper leaves",
        ],
        "prevent": [
            "Rotate crops for at least 2 years",
            "Mulch under the plants and avoid overhead watering",
            "Clear old tomato debris at the end of the season",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Spider mites",
        "kind": "Pest",
        "urgency": "Medium",
        "cause": "Pest (two-spotted spider mite)",
        "summary": "Tiny pests that cause pale speckles and fine webbing, mostly in hot, dry weather.",
        "symptoms": [
            "Fine pale yellow speckles on the leaf surface",
            "Leaves turn bronze or dry out and may drop",
            "Thin webbing on the underside of leaves and between stems",
        ],
        "conditions": "Hot, dry, dusty conditions. Populations grow very quickly in warm weather.",
        "act": [
            "Check the underside of leaves, where mites hide",
            "Rinse plants with a firm stream of water to knock mites off",
            "Use insecticidal soap or neem as directed, and ask local experts about a miticide for severe cases",
        ],
        "prevent": [
            "Keep plants watered and avoid drought stress",
            "Inspect plants weekly in hot weather",
            "Encourage natural predators such as ladybirds and predatory mites",
        ],
    },
    "Tomato___Target_Spot": {
        "name": "Target spot",
        "kind": "Fungal",
        "urgency": "Medium",
        "cause": "Fungus (Corynespora cassiicola)",
        "summary": "Brown spots with rings, found on leaves and sometimes on fruit.",
        "symptoms": [
            "Small brown spots that grow into rings with a yellow border",
            "Spots can join together and kill large areas of the leaf",
            "Sunken dark spots on fruit in serious cases",
        ],
        "conditions": "Warm temperatures with long periods of leaf wetness. Dense plant growth makes it worse.",
        "act": [
            "Remove affected leaves and plant debris",
            "Open the canopy so leaves dry faster",
            "Ask your local agriculture office about fungicide if it keeps spreading",
        ],
        "prevent": [
            "Space and stake plants for good airflow",
            "Avoid overhead irrigation",
            "Rotate crops and clean up plant residue after harvest",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato yellow leaf curl virus",
        "kind": "Viral",
        "urgency": "High",
        "cause": "Virus spread by whiteflies",
        "summary": "Leaves curl upward and turn yellow while the plant stays small and bushy.",
        "symptoms": [
            "Leaves curl upward and inward, with yellow edges",
            "Plants are stunted and look bushy",
            "Flowers drop and very little fruit forms",
        ],
        "conditions": "Spread by whiteflies feeding on infected plants and then healthy ones. Most common in warm weather.",
        "act": [
            "There is no cure: remove and bag infected plants so they do not feed whiteflies",
            "Check the underside of leaves for tiny white flying insects",
            "Control whiteflies with sticky traps and ask local experts about suitable sprays",
        ],
        "prevent": [
            "Use resistant or tolerant varieties where available",
            "Protect seedlings with fine insect netting",
            "Remove weeds that host whiteflies",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato mosaic virus",
        "kind": "Viral",
        "urgency": "Medium",
        "cause": "Virus (spread by touch, tools and seed)",
        "summary": "Mottled light and dark green leaves, sometimes narrow or twisted.",
        "symptoms": [
            "Mosaic pattern of light and dark green patches on leaves",
            "Leaves can be narrow, curled or fern-like",
            "Fruit may ripen unevenly or have brown patches inside",
        ],
        "conditions": "The virus spreads on hands, tools and clothing, and can survive for a long time on surfaces and in plant debris.",
        "act": [
            "There is no cure: remove infected plants carefully",
            "Wash hands with soap and clean tools after handling plants",
            "Do not handle healthy plants right after touching infected ones",
        ],
        "prevent": [
            "Use certified virus-free seed and resistant varieties",
            "Disinfect tools and stakes between seasons",
            "Remove tomato debris and weeds from the garden",
        ],
    },
    "Tomato___healthy": {
        "name": "Healthy leaf",
        "kind": "Healthy",
        "urgency": "Low",
        "cause": "No disease detected",
        "summary": "The leaf looks healthy to the model.",
        "symptoms": [
            "Even green color without spots or patches",
            "Flat leaves with normal shape",
            "No webbing, mold or insects",
        ],
        "conditions": "Healthy plants still need regular watching, since diseases can start quickly in humid weather.",
        "act": [
            "Keep up regular watering at the base of the plant",
            "Check leaves, including the undersides, once a week",
            "Keep the area clean of fallen leaves",
        ],
        "prevent": [
            "Rotate crops every season",
            "Give plants space and good airflow",
            "Use clean tools and healthy seedlings",
        ],
    },
}

CARE_GUIDE = [
    {
        "title": "Sun and location",
        "lead": "Tomatoes want 6 to 8 hours of direct sun a day.",
        "points": [
            "Pick a spot with morning sun and good airflow",
            "Warm days of 21 to 29 °C are ideal; very hot days (above about 32 °C) can make flowers drop",
            "Avoid planting where tomatoes, potatoes or peppers grew last season",
        ],
    },
    {
        "title": "Soil",
        "lead": "Loose, well-drained soil rich in compost works best.",
        "points": [
            "Aim for a slightly acidic soil, pH 6.0 to 6.8",
            "Mix in well-rotted compost or manure before planting",
            "In pots, use at least a 20 litre container with drainage holes",
        ],
    },
    {
        "title": "Planting and spacing",
        "lead": "Give each plant room, because crowded plants stay wet and get sick.",
        "points": [
            "Space plants about 45 to 60 cm apart, with rows about 75 to 90 cm apart",
            "Bury the stem deeper than the original soil level; it will grow extra roots",
            "Harden off seedlings for a week before moving them outside",
        ],
    },
    {
        "title": "Watering",
        "lead": "Water deeply and regularly, at the base of the plant.",
        "points": [
            "Most plants need about 2.5 cm of water a week, more in hot weather",
            "Water in the morning so leaves dry quickly during the day",
            "Uneven watering causes blossom end rot and cracked fruit",
        ],
    },
    {
        "title": "Feeding",
        "lead": "Feed lightly early on and shift to a fruiting feed once flowers appear.",
        "points": [
            "Too much nitrogen gives lots of leaves and few tomatoes",
            "Use a feed that is higher in phosphorus and potassium once fruit sets",
            "Follow the dose on the label and water after feeding",
        ],
    },
    {
        "title": "Support and pruning",
        "lead": "Keeping plants off the ground reduces disease.",
        "points": [
            "Stake or cage plants early, before they flop over",
            "Remove the lowest leaves that touch the soil",
            "On indeterminate (vining) types, pinch out side shoots for better airflow",
        ],
    },
    {
        "title": "Mulch and cleanliness",
        "lead": "A clean bed means fewer diseases.",
        "points": [
            "Mulch with straw or dry leaves to stop soil from splashing on leaves",
            "Clean tools and stakes between plants and between seasons",
            "Remove fallen leaves and fruit, and dispose of them instead of composting diseased material",
        ],
    },
    {
        "title": "Crop rotation",
        "lead": "Do not grow tomatoes in the same place year after year.",
        "points": [
            "Wait 2 to 3 years before returning tomatoes to the same bed",
            "Potatoes, peppers and brinjal (eggplant) share many tomato diseases",
            "Follow with beans, leafy greens or onions",
        ],
    },
]

PROBLEMS_NOT_DISEASE = [
    ("Blossom end rot", "A dark, sunken patch at the bottom of the fruit. It is caused by uneven watering that stops calcium reaching the fruit, not by a germ."),
    ("Flowers dropping", "Often caused by very hot days, cold nights or too little water. Plants usually recover when the weather settles."),
    ("Cracked fruit", "Heavy watering after a dry spell makes the fruit swell too quickly. Keep watering steady and use mulch."),
    ("Yellow lower leaves", "Old leaves naturally yellow. If it climbs quickly, check watering, nutrients and signs of disease."),
]

WEEKLY_ROUTINE = [
    "Look at both sides of the leaves, especially the lower ones",
    "Remove yellow or spotted leaves and any fallen fruit",
    "Tie in new growth and check the stakes",
    "Check the soil moisture before you water",
]

# --------------------------------------------------
# STYLING
# --------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #10231A;
  --forest: #0E2C1F;
  --moss: #2E6A3E;
  --mint: #CFE3D3;
  --paper: #F5F7F4;
  --card: #FFFFFF;
  --line: #DCE3DC;
  --muted: #5B6B61;
  --accent: #D8402F;
}

html, body, .stApp {
  background: var(--paper) !important;
  color: var(--ink);
  font-family: 'Instrument Sans', system-ui, sans-serif;
}
.stApp p, .stApp li, .stApp span, .stApp label, .stApp div { color: inherit; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .ag-display {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 500;
  color: var(--ink);
  letter-spacing: -0.015em;
}

#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stSidebar"],
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }

.block-container {
  max-width: 1120px;
  padding: 1.2rem 1.6rem 3rem 1.6rem !important;
}

/* top bar */
.ag-brand { display:flex; align-items:center; gap:.65rem; font-family:'Fraunces',Georgia,serif;
  font-weight:600; font-size:1.4rem; color:var(--forest); letter-spacing:-0.01em; }
.ag-rule { height:1px; background:var(--line); margin:.1rem 0 2rem 0; }

.st-key-nav [data-testid^="stBaseButton"], .st-key-nav button {
  background: transparent !important; border: none !important; border-radius: 0 !important;
  border-bottom: 2px solid transparent !important; box-shadow: none !important;
  padding: .55rem .1rem !important; min-height: 0 !important;
}
.st-key-nav [data-testid^="stBaseButton"] p, .st-key-nav button p {
  color: var(--muted) !important; font-weight: 500; font-size: .97rem;
}
.st-key-nav [data-testid^="stBaseButton"]:hover p, .st-key-nav button:hover p { color: var(--ink) !important; }
.st-key-nav [data-testid="stBaseButton-primary"], .st-key-nav button[kind="primary"] {
  border-bottom: 2px solid var(--ink) !important;
}
.st-key-nav [data-testid="stBaseButton-primary"] p, .st-key-nav button[kind="primary"] p {
  color: var(--ink) !important; font-weight: 600;
}

/* hero */
.ag-hero-text h1 { font-size: clamp(2.2rem, 4.6vw, 3.5rem); line-height: 1.08; margin: .2rem 0 1.1rem 0; max-width: 15ch; }
.ag-hero-text p { color: var(--muted) !important; font-size: 1.1rem; line-height: 1.6; max-width: 44ch; margin: 0 0 1.4rem 0; }
.ag-panel { background: radial-gradient(circle at 30% 20%, #1D4E37 0%, #0E2C1F 72%); border-radius: 20px; padding: 1.2rem 1.2rem .6rem 1.2rem; }
.ag-panel svg { width: 100%; height: auto; display: block; }
.ag-cap { color: #9DB8A5 !important; font-size: .8rem; text-align: right; margin: .2rem .2rem .4rem 0; }

.st-key-cta1 [data-testid^="stBaseButton"], .st-key-cta1 button {
  background: var(--forest) !important; border: 1px solid var(--forest) !important;
  border-radius: 10px !important; padding: .7rem 1.2rem !important;
}
.st-key-cta1 button p, .st-key-cta1 [data-testid^="stBaseButton"] p { color: #fff !important; font-weight: 600; }
.st-key-cta1 button:hover { background: var(--moss) !important; border-color: var(--moss) !important; }
.st-key-cta2 [data-testid^="stBaseButton"], .st-key-cta2 button {
  background: transparent !important; border: 1px solid var(--ink) !important;
  border-radius: 10px !important; padding: .7rem 1.2rem !important;
}
.st-key-cta2 button p, .st-key-cta2 [data-testid^="stBaseButton"] p { color: var(--ink) !important; font-weight: 600; }
.st-key-cta2 button:hover { background: var(--mint) !important; }

.ag-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 2.6rem 0 .4rem 0; }
.ag-stats > div { border-top: 1px solid var(--ink); padding-top: .8rem; }
.ag-stats b { display: block; font-family: 'Fraunces', Georgia, serif; font-weight: 500; font-size: 2rem; letter-spacing: -0.02em; }
.ag-stats span { color: var(--muted) !important; font-size: .92rem; }

/* section headings */
.ag-h2 { font-family:'Fraunces',Georgia,serif; font-weight:500; font-size:1.9rem; letter-spacing:-0.015em; margin:2.8rem 0 .4rem 0; }
.ag-sub { color:var(--muted) !important; margin:0 0 1.2rem 0; max-width:62ch; line-height:1.6; }

/* how it works */
.ag-step2 { border-top: 1px solid var(--line); padding-top: 1rem; }
.ag-step2 .n { font-family:'Fraunces',Georgia,serif; font-size:2.4rem; color:var(--moss) !important; line-height:1; }
.ag-step2 h4 { margin: .6rem 0 .3rem 0; font-size:1.2rem; }
.ag-step2 p { color:var(--muted) !important; margin:0; line-height:1.55; }

/* cards */
.ag-card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:1.2rem 1.3rem; }
.ag-card h4 { margin:0 0 .4rem 0; font-size:1.2rem; }
.ag-card p { margin:0; color:var(--muted) !important; line-height:1.55; font-size:.97rem; }
.ag-card ul { margin:.5rem 0 0 0; padding-left:1.1rem; color:var(--ink); }
.ag-card li { margin:.3rem 0; line-height:1.5; font-size:.96rem; }
.ag-kind { font-size:.82rem; font-weight:600; }

/* tags */
.ag-tag { display:inline-block; font-size:.78rem; font-weight:600; padding:.12rem .55rem; border-radius:6px; margin-right:.5rem; }
.ag-urg { display:inline-flex; align-items:center; gap:.35rem; font-size:.82rem; color:var(--muted) !important; }
.ag-urg i { width:.55rem; height:.55rem; border-radius:50%; display:inline-block; }

/* result card */
.ag-result { background:var(--card); border:1px solid var(--line); border-top:5px solid var(--moss); border-radius:12px; padding:1.4rem 1.5rem; }
.ag-result h2 { margin:.5rem 0 .3rem 0; font-size:2rem; }
.ag-bar { background:#E7ECE6; border-radius:999px; height:.5rem; overflow:hidden; margin:.35rem 0 1rem 0; }
.ag-bar > span { display:block; height:100%; background:var(--moss); border-radius:999px; }
.ag-row { display:flex; justify-content:space-between; font-size:.92rem; color:var(--muted); }
.ag-note { background:#FBF3DE; border:1px solid #EBD79F; color:#5E4410 !important; border-radius:10px; padding:.8rem 1rem; margin:.8rem 0; font-size:.95rem; line-height:1.5; }
.ag-note.bad { background:#FBE8E4; border-color:#EDB5AC; color:#7A2318 !important; }

/* footer */
.ag-footer { border-top:1px solid var(--line); margin-top:3.5rem; padding-top:1.1rem; color:var(--muted) !important;
  font-size:.85rem; display:flex; justify-content:space-between; flex-wrap:wrap; gap:.6rem; }

/* uploader and expanders */
[data-testid="stFileUploader"] section { background:var(--card); border:1.5px dashed var(--moss); border-radius:12px; }
[data-testid="stExpander"] { background:var(--card); border:1px solid var(--line) !important; border-radius:12px; }
[data-testid="stExpander"] summary p { font-weight:600; }
button[data-baseweb="tab"] p { font-weight:600; }

@media (max-width: 700px) {
  .ag-stats { grid-template-columns: 1fr; gap: 1rem; }
  .block-container { padding: 1rem .9rem 3rem .9rem !important; }
}
</style>
"""

BRAND_HTML = (
    '<div class="ag-brand">'
    '<svg width="30" height="30" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect width="32" height="32" rx="9" fill="#0E2C1F"/>'
    '<path d="M16 26C9 22 8 14 12 10C14 8 15 7 16 5C17 7 18 8 20 10C24 14 23 22 16 26Z" fill="#7FBF93"/>'
    '<path d="M16 25V11" stroke="#0E2C1F" stroke-width="1.4"/>'
    '<circle cx="23" cy="9" r="3" fill="#D8402F"/></svg>AgriShield</div>'
)

HERO_SVG = (
    '<svg viewBox="0 0 440 460" xmlns="http://www.w3.org/2000/svg" role="img" '
    'aria-label="A tomato leaf being scanned, with a disease spot marked">'
    '<defs><linearGradient id="lf" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3F8A5A"/>'
    '<stop offset="1" stop-color="#1F5236"/></linearGradient>'
    '<linearGradient id="sc" x1="0" x2="1"><stop offset="0" stop-color="#7FBF93" stop-opacity="0"/>'
    '<stop offset=".5" stop-color="#7FBF93"/><stop offset="1" stop-color="#7FBF93" stop-opacity="0"/></linearGradient></defs>'
    '<g stroke="#CFE3D3" stroke-width="2.5" fill="none" stroke-linecap="round">'
    '<path d="M40 90V50H80"/><path d="M360 50H400V90"/><path d="M40 370V410H80"/><path d="M360 410H400V370"/></g>'
    '<path d="M200 395C112 345 86 236 140 150C166 108 190 88 200 56C210 88 234 108 260 150C314 236 288 345 200 395Z" '
    'fill="url(#lf)" stroke="#8FC79F" stroke-width="2"/>'
    '<path d="M200 395V66" stroke="#A9D6B6" stroke-width="2.5"/>'
    '<g stroke="#A9D6B6" stroke-width="1.6" stroke-linecap="round" opacity=".8">'
    '<path d="M200 330L146 284"/><path d="M200 330L254 284"/><path d="M200 266L140 218"/>'
    '<path d="M200 266L260 218"/><path d="M200 200L158 160"/><path d="M200 200L242 160"/></g>'
    '<circle cx="236" cy="268" r="26" fill="none" stroke="#E0B867" stroke-width="2" stroke-dasharray="4 4"/>'
    '<circle cx="236" cy="268" r="14" fill="#6E4A1F"/><circle cx="236" cy="268" r="6" fill="#3E2A10"/>'
    '<circle cx="150" cy="232" r="7" fill="#6E4A1F" opacity=".85"/>'
    '<rect x="60" y="205" width="320" height="2" fill="url(#sc)"/>'
    '<path d="M262 268H318" stroke="#E7F1E8" stroke-width="1.6"/><circle cx="262" cy="268" r="4" fill="#D8402F"/>'
    '<rect x="318" y="244" width="104" height="48" rx="10" fill="#F5F7F4"/>'
    '<text x="332" y="264" font-family="Instrument Sans, sans-serif" font-size="13" font-weight="600" fill="#10231A">Early blight</text>'
    '<text x="332" y="282" font-family="Instrument Sans, sans-serif" font-size="12" fill="#5B6B61">91% confidence</text>'
    '</svg>'
)

FOOTER_HTML = (
    '<div class="ag-footer"><span>AgriShield is an educational student project.</span>'
    '<span>Results are AI predictions, not a diagnosis. Check with a local agriculture expert before treating plants.</span></div>'
)


def tag(kind):
    fg, bg = KIND_COLORS[kind]
    return f'<span class="ag-tag" style="color:{fg};background:{bg}">{kind}</span>'


def urgency(level):
    return (
        f'<span class="ag-urg"><i style="background:{URGENCY_COLORS[level]}"></i>'
        f'{level} urgency</span>'
    )


def bullets(items):
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def heading(title, sub=""):
    html = f'<div class="ag-h2">{title}</div>'
    if sub:
        html += f'<p class="ag-sub">{sub}</p>'
    st.markdown(html, unsafe_allow_html=True)


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
# PREDICTION
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
    """Returns (ranked tomato results, tomato_share).

    ranked: list of (class_name, probability) among tomato classes, best first.
    tomato_share: how much of the model's total probability sits on tomato
    classes. A low value suggests the photo may not be a tomato leaf.
    """
    processed_image = prepare_image(image)
    predictions = model.predict(processed_image, verbose=0)[0]

    tomato_probs = predictions[TOMATO_IDX]
    tomato_share = float(tomato_probs.sum())
    tomato_probs = tomato_probs / tomato_probs.sum()

    order = np.argsort(tomato_probs)[::-1]
    ranked = [(CLASS_NAMES[TOMATO_IDX[i]], float(tomato_probs[i])) for i in order]

    return ranked, tomato_share


def result_card(ranked, tomato_share):
    best_class, confidence = ranked[0]
    info = TOMATO_DISEASES[best_class]
    color = URGENCY_COLORS[info["urgency"]] if best_class != "Tomato___healthy" else "#2E6A3E"

    notes = ""
    if tomato_share < 0.5:
        notes += (
            '<div class="ag-note bad">This photo may not be a tomato leaf, so the result '
            'is unreliable. Try a close, well-lit photo of one tomato leaf.</div>'
        )
    elif confidence < 0.6:
        notes += (
            '<div class="ag-note">The model is not very sure about this one. '
            'Try another photo, or compare with the guide in the Disease guide page.</div>'
        )

    others = ""
    for name, p in ranked[1:3]:
        others += (
            f'<div class="ag-row"><span>{TOMATO_DISEASES[name]["name"]}</span>'
            f'<span>{p * 100:.1f}%</span></div>'
            f'<div class="ag-bar"><span style="width:{p * 100:.1f}%;background:#9DB59A"></span></div>'
        )

    html = (
        f'<div class="ag-result" style="border-top-color:{color}">'
        f'{tag(info["kind"])}{urgency(info["urgency"])}'
        f'<h2>{info["name"]}</h2>'
        f'<p class="ag-sub" style="margin-bottom:.9rem">{info["summary"]}</p>'
        f'<div class="ag-row"><span>Model confidence</span><span><b>{confidence * 100:.1f}%</b></span></div>'
        f'<div class="ag-bar"><span style="width:{confidence * 100:.1f}%;background:{color}"></span></div>'
        f'{notes}'
        f'<h4 style="margin:.6rem 0 .1rem 0">What to do now</h4>{bullets(info["act"])}'
        f'<h4 style="margin:1rem 0 .1rem 0">How to prevent it</h4>{bullets(info["prevent"])}'
        f'<h4 style="margin:1rem 0 .4rem 0">Other possibilities</h4>{others}'
        f'<p style="color:#55645A;font-size:.85rem;margin-top:.8rem">This is an AI prediction, not a confirmed diagnosis. '
        f'Check the plant and ask a local agriculture expert before using any treatment.</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
PAGES = ["Home", "Check a leaf", "Disease guide", "Care guide", "About"]

if "page_choice" not in st.session_state:
    st.session_state["page_choice"] = "Home"


def go_to(page):
    st.session_state["page_choice"] = page


st.markdown(CSS, unsafe_allow_html=True)

with st.container(key="nav"):
    nav_cols = st.columns([2.4, 1, 1.25, 1.25, 1.1, 0.9], vertical_alignment="center")
    with nav_cols[0]:
        st.markdown(BRAND_HTML, unsafe_allow_html=True)
    for col, name in zip(nav_cols[1:], PAGES):
        with col:
            st.button(
                name,
                key=f"nav_{name}",
                on_click=go_to,
                args=(name,),
                type="primary" if st.session_state["page_choice"] == name else "secondary",
                use_container_width=True,
            )

st.markdown('<div class="ag-rule"></div>', unsafe_allow_html=True)

page = st.session_state["page_choice"]

# --------------------------------------------------
# HOME
# --------------------------------------------------
if page == "Home":
    hero_left, hero_right = st.columns([1.1, 1], gap="large", vertical_alignment="center")

    with hero_left:
        st.markdown(
            '<div class="ag-hero-text"><h1>Spot tomato leaf disease from a single photo</h1>'
            '<p>Upload a photo of a leaf. AgriShield names the most likely problem from nine common tomato '
            'diseases and pests, then explains what to do next and how to prevent it.</p></div>',
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2)
        with b1:
            st.button("Check a leaf", key="cta1", on_click=go_to, args=("Check a leaf",), use_container_width=True)
        with b2:
            st.button("Read the disease guide", key="cta2", on_click=go_to, args=("Disease guide",), use_container_width=True)

    with hero_right:
        st.markdown(
            f'<div class="ag-panel">{HERO_SVG}<p class="ag-cap">Example of a result</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="ag-stats">'
        '<div><b>9</b><span>tomato diseases and pests covered</span></div>'
        '<div><b>70,000+</b><span>leaf images used to train the model</span></div>'
        '<div><b>93%+</b><span>accuracy on held-out validation images</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    heading("How it works")
    steps = [
        ("1", "Take a photo", "Take a close, well-lit photo of one leaf against a plain background."),
        ("2", "Upload it", "Add the photo on the Check a leaf page, or use your phone camera."),
        ("3", "Read the result", "See the likely disease, how sure the model is, and what to do about it."),
    ]
    cols = st.columns(3, gap="large")
    for col, (n, title, text) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="ag-step2"><div class="n">{n}</div><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True,
            )

    heading(
        "What AgriShield can recognise",
        "Open the Disease guide for causes, symptoms, treatment and prevention.",
    )
    items = [(k, v) for k, v in TOMATO_DISEASES.items() if k != "Tomato___healthy"]
    for row in range(0, len(items), 3):
        cols = st.columns(3, gap="large")
        for col, (_, info) in zip(cols, items[row:row + 3]):
            fg, _bg = KIND_COLORS[info["kind"]]
            with col:
                st.markdown(
                    f'<div class="ag-card" style="border-top:3px solid {fg}">'
                    f'<span class="ag-kind" style="color:{fg}">{info["kind"]}</span>'
                    f'<h4 style="margin-top:.35rem">{info["name"]}</h4>'
                    f'<p>{info["summary"]}</p></div>',
                    unsafe_allow_html=True,
                )
        st.write("")

# --------------------------------------------------
# CHECK A LEAF
# --------------------------------------------------
elif page == "Check a leaf":
    heading(
        "Check a tomato leaf",
        "Upload one clear photo of a single tomato leaf. The result appears next to your photo.",
    )

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        tab_upload, tab_camera = st.tabs(["Upload a photo", "Use camera"])
        with tab_upload:
            uploaded_file = st.file_uploader(
                "Choose a tomato leaf image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
        with tab_camera:
            camera_file = st.camera_input("Take a photo of the leaf", label_visibility="collapsed")

        source = uploaded_file if uploaded_file is not None else camera_file

        with st.expander("Tips for a good photo"):
            st.markdown(
                "- Fill the frame with one leaf and keep it in focus\n"
                "- Use daylight and avoid harsh shadows\n"
                "- Use a plain background if you can\n"
                "- Take a second photo of the underside if you see webbing or mold"
            )

        if source is not None:
            image = Image.open(source).convert("RGB")
            st.image(image, caption="Your photo", use_container_width=True)

    with right:
        if source is None:
            st.markdown(
                '<div class="ag-card"><h4>Your result will appear here</h4>'
                '<p>Add a photo on the left to see the likely disease, how confident the model is '
                'and what to do next.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            try:
                with st.spinner("Checking the leaf..."):
                    model = load_model()
                    ranked, tomato_share = predict_disease(model, image)
                result_card(ranked, tomato_share)
            except Exception as error:
                st.error(
                    "The photo could not be checked. Try a different image, "
                    "or open the app logs if this keeps happening."
                )
                st.exception(error)

# --------------------------------------------------
# DISEASE GUIDE
# --------------------------------------------------
elif page == "Disease guide":
    heading(
        "Tomato disease guide",
        "Nine common problems on tomato leaves: what causes them, what they look like, and what to do. "
        "Open any card for the full details. Urgency shows how quickly to act: low can wait for routine care, high needs action within days.",
    )

    kinds = ["All", "Fungal", "Bacterial", "Viral", "Pest"]
    choice = st.segmented_control("Filter by cause", kinds, default="All", key="kind_filter") or "All"

    shown = [
        (k, v) for k, v in TOMATO_DISEASES.items()
        if k != "Tomato___healthy" and (choice == "All" or v["kind"] == choice)
    ]

    for row in range(0, len(shown), 2):
        cols = st.columns(2, gap="large")
        for col, (_, info) in zip(cols, shown[row:row + 2]):
            with col:
                st.markdown(
                    f'<div class="ag-card">{tag(info["kind"])}{urgency(info["urgency"])}'
                    f'<h4 style="margin-top:.6rem">{info["name"]}</h4>'
                    f'<p>{info["summary"]}</p>'
                    f'<p style="margin-top:.5rem;font-size:.88rem"><b>Cause:</b> {info["cause"]}</p></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Symptoms, treatment and prevention"):
                    st.markdown("**What it looks like**")
                    st.markdown("\n".join(f"- {s}" for s in info["symptoms"]))
                    st.markdown("**When it appears**")
                    st.write(info["conditions"])
                    st.markdown("**What to do**")
                    st.markdown("\n".join(f"- {s}" for s in info["act"]))
                    st.markdown("**How to prevent it**")
                    st.markdown("\n".join(f"- {s}" for s in info["prevent"]))
        st.write("")

    st.markdown(
        '<div class="ag-note">Always confirm a diagnosis with your local agriculture office before spraying. '
        'The right product and dose depend on your region, crop stage and local rules.</div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# CARE GUIDE
# --------------------------------------------------
elif page == "Care guide":
    heading(
        "How to look after tomato plants",
        "Healthy, well-spaced plants that get steady water are far less likely to catch the diseases above.",
    )

    for row in range(0, len(CARE_GUIDE), 2):
        cols = st.columns(2, gap="large")
        for col, item in zip(cols, CARE_GUIDE[row:row + 2]):
            with col:
                st.markdown(
                    f'<div class="ag-card"><h4>{item["title"]}</h4>'
                    f'<p style="color:#17241B;font-weight:500">{item["lead"]}</p>'
                    f'{bullets(item["points"])}</div>',
                    unsafe_allow_html=True,
                )
        st.write("")

    heading(
        "Problems that look like disease but are not",
        "These are caused by weather or watering. They do not spread from plant to plant.",
    )
    cols = st.columns(2, gap="large")
    for i, (title, text) in enumerate(PROBLEMS_NOT_DISEASE):
        with cols[i % 2]:
            st.markdown(
                f'<div class="ag-card"><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True,
            )
            st.write("")

    heading("A 5-minute weekly check")
    st.markdown(
        f'<div class="ag-card">{bullets(WEEKLY_ROUTINE)}</div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# ABOUT
# --------------------------------------------------
elif page == "About":
    heading("About AgriShield")
    st.markdown(
        '<p class="ag-sub">AgriShield is a student project that uses a convolutional neural network (CNN) '
        'to help recognise common tomato leaf diseases from a photo. It was built by Gauri as an educational prototype.</p>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            '<div class="ag-card"><h4>How it works</h4>'
            + bullets([
                "Your photo is resized to 128 × 128 pixels",
                "The CNN, trained on about 70,000 labelled leaf images across 38 plant classes, scores every class",
                "The app keeps only the tomato classes and shows the most likely ones",
                "A short guide to symptoms, care and prevention is shown with each result",
            ])
            + '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="ag-card"><h4>Technology</h4>'
            + bullets([
                "Python, TensorFlow and Keras for the model",
                "NumPy and Pillow for image handling",
                "Streamlit for the website",
                "Model file hosted on Google Drive and downloaded on first use",
            ])
            + '</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        '<div class="ag-note"><b>Limitations.</b> The model was trained mostly on clear, close-up leaf photos '
        'with plain backgrounds, so photos with busy backgrounds or poor light can give wrong answers. '
        'It only covers the nine tomato conditions listed in the Disease guide, and it is a first check, '
        'not a diagnosis. Please confirm with an agriculture expert before treating plants.</div>',
        unsafe_allow_html=True,
    )

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
