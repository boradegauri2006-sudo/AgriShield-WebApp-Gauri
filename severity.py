
import cv2
import numpy as np
from PIL import Image


def estimate_leaf_damage(image):
    """
    Experimental colour-based indicator.
    This does NOT measure confirmed disease severity.
    Input: a PIL Image.
    """

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    rgb = np.array(image.convert("RGB"))

    # Convert RGB image to HSV colour space.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Approximate green-colour range.
    lower_green = np.array([25, 35, 25])
    upper_green = np.array([95, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    total_pixels = green_mask.size

    if total_pixels == 0:
        return {
            "percentage": 0.0,
            "label": "Unable to estimate"
        }

    green_pixels = cv2.countNonZero(green_mask)

    green_percentage = (
        green_pixels / total_pixels
    ) * 100

    non_green_percentage = (
        100 - green_percentage
    )

    return {
        "percentage": round(
            non_green_percentage, 1
        ),
        "label": (
            "Experimental non-green colour estimate; "
            "not a validated damage percentage."
        )
    }
