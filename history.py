
import json
from pathlib import Path
from datetime import datetime

HISTORY_FILE = Path("diagnosis_history.json")


def load_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        with HISTORY_FILE.open(
            "r", encoding="utf-8"
        ) as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return []


def save_diagnosis(
    disease,
    confidence,
    severity_percentage
):
    history = load_history()

    record = {
        "date": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "disease": str(disease),
        "confidence": round(
            float(confidence), 2
        ),
        "severity_estimate": round(
            float(severity_percentage), 1
        )
    }

    history.insert(0, record)

    # Keep the latest 100 records.
    history = history[:100]

    with HISTORY_FILE.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            history,
            file,
            ensure_ascii=False,
            indent=2
        )

    return history
