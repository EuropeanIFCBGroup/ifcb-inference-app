"""What the viewer shows versus what the model is given.

Small IFCB ROIs are only tens of pixels across, so the viewer enlarges them to
stay legible. That enlargement is a display concern and must not reach the
model: nearest-neighbour upscaling before the square-pad and resize produces a
different tensor, and on real ROIs it changed the top class often enough that the
app disagreed with itself — and with the pipeline — depending on whether a
prediction came from the upload or from pressing Classify afterwards.

Keeping both decisions here, free of any UI framework, is what lets them be
tested directly.
"""

from PIL import Image

MAX_DISPLAY_SCALE = 3
DISPLAY_TARGET = 400


def display_image(image):
    """Enlarge a small image for display, capped at :data:`MAX_DISPLAY_SCALE`.

    Nearest-neighbour, so pixels stay crisp rather than smeared. For viewing
    only — pair it with :func:`image_to_classify`, never feed the result to a
    model.
    """
    if image is None:
        return image
    width, height = image.size
    scale = min(MAX_DISPLAY_SCALE, DISPLAY_TARGET / max(width, height, 1))
    if scale > 1:
        return image.resize((int(width * scale), int(height * scale)), Image.NEAREST)
    return image


def image_to_classify(original, shown):
    """Pick the image to score: the one that arrived, not the one on display.

    ``shown`` is consulted only for whether the viewer still holds an image, so
    clearing it clears the prediction. It is scored itself only as a fallback,
    when no original was ever captured — a prediction from the displayed copy
    beats none at all.
    """
    if shown is None:
        return None
    return original if original is not None else shown
