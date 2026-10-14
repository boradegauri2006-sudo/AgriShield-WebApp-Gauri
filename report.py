
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet


def create_diagnosis_pdf(
    disease,
    confidence,
    severity,
    actions,
    language="English"
):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    story = []

    if language == "मराठी":
        title = "अ‍ॅग्रीशील्ड - रोग निदान अहवाल"
        disease_label = "ओळखलेला रोग"
        confidence_label = "अंदाजाची खात्री"
        severity_label = "दृश्य नुकसान अंदाज"
        actions_label = "सुचवलेले उपाय"
    else:
        title = "AgriShield - Diagnosis Report"
        disease_label = "Predicted disease"
        confidence_label = "Prediction confidence"
        severity_label = "Visual colour estimate"
        actions_label = "Recommended actions"

    story.append(
        Paragraph(escape(title), styles["Title"])
    )
    story.append(Spacer(1, 18))

    details = [
        (disease_label, disease),
        (confidence_label, f"{float(confidence):.2f}%"),
        (severity_label, f"{float(severity):.1f}%")
    ]

    for label, value in details:
        text = (
            f"<b>{escape(label)}:</b> "
            f"{escape(str(value))}"
        )
        story.append(
            Paragraph(text, styles["BodyText"])
        )
        story.append(Spacer(1, 10))

    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            escape(actions_label),
            styles["Heading2"]
        )
    )

    for action in actions:
        story.append(
            Paragraph(
                "- " + escape(str(action)),
                styles["BodyText"]
            )
        )
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 18))

    disclaimer = (
        "AI-assisted screening result. Confirm uncertain "
        "diagnoses with an agricultural expert. The colour "
        "estimate is experimental and is not a validated "
        "measurement of disease severity."
    )

    if language == "मराठी":
        disclaimer = (
            "हा AI-आधारित प्राथमिक अंदाज आहे. निदानाबाबत "
            "शंका असल्यास कृषी तज्ज्ञांचा सल्ला घ्या. "
            "रंगावर आधारित नुकसान अंदाज प्रायोगिक आहे."
        )

    story.append(
        Paragraph(
            escape(disclaimer),
            styles["Italic"]
        )
    )

    document.build(story)

    return buffer.getvalue()
