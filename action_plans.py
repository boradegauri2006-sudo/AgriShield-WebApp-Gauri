
ACTION_PLANS = {
    "Tomato___Early_blight": {
        "English": [
            "Inspect leaves regularly for spreading spots.",
            "Remove badly affected leaves when practical.",
            "Avoid overhead watering where possible.",
            "Keep adequate spacing for airflow.",
            "Seek local agricultural advice if symptoms spread."
        ],
        "मराठी": [
            "पानांवरील डाग वाढत आहेत का ते तपासा.",
            "शक्य असल्यास जास्त बाधित पाने काढा.",
            "शक्यतो पानांवरून पाणी मारणे टाळा.",
            "हवेचा प्रवाह राहील असे अंतर ठेवा.",
            "लक्षणे वाढल्यास कृषी तज्ज्ञांचा सल्ला घ्या."
        ]
    },

    "Tomato___Late_blight": {
        "English": [
            "Check plants regularly for rapidly spreading lesions.",
            "Avoid handling plants when foliage is wet.",
            "Remove affected plant material carefully.",
            "Avoid overhead watering where possible.",
            "Ask a local agricultural expert for timely advice."
        ],
        "मराठी": [
            "रोगाची लक्षणे वेगाने वाढत आहेत का ते तपासा.",
            "झाडांची पाने ओली असताना त्यांना हाताळणे टाळा.",
            "बाधित भाग काळजीपूर्वक काढा.",
            "शक्यतो पानांवरून पाणी मारणे टाळा.",
            "वेळीच कृषी तज्ज्ञांचा सल्ला घ्या."
        ]
    },

    "Healthy": {
        "English": [
            "Continue regular plant inspections.",
            "Water appropriately and maintain good airflow.",
            "Monitor the plant as it grows."
        ],
        "मराठी": [
            "झाडांची नियमित तपासणी करा.",
            "योग्य प्रमाणात पाणी द्या आणि हवा खेळती ठेवा.",
            "झाडाच्या वाढीवर लक्ष ठेवा."
        ]
    }
}


def get_action_plan(class_name, language="English"):
    """
    Return actions for a model class name.
    Fall back to English if a translation is missing.
    """

    disease = ACTION_PLANS.get(class_name)

    if disease is None:
        return (
            ["Consult a local agricultural expert."]
            if language == "English"
            else ["स्थानिक कृषी तज्ज्ञांचा सल्ला घ्या."]
        )

    return disease.get(
        language,
        disease.get("English", [])
    )
