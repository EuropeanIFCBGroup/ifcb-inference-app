"""What gets classified must be the image as it arrived, not the one on display.

The viewer holds a nearest-neighbour upscale so small ROIs stay legible. Feeding
that copy to the model changes the tensor it sees — on real ROIs it moved the top
class often enough to make the app disagree with itself, and with the pipeline,
depending on whether a prediction came from the upload or from pressing Classify
afterwards.
"""

import asyncio
import shutil

import numpy as np
import pytest
from PIL import Image

import model as app_model
from viewer import display_image, image_to_classify


@pytest.fixture
def roi():
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 255, (41, 53), dtype=np.uint8)).convert("RGB")


@pytest.fixture
def model_name():
    if not app_model.AVAILABLE_MODELS:
        pytest.skip("no model installed under data/models")
    return app_model.DEFAULT_MODEL


def test_display_upscale_changes_the_image(roi):
    """The premise: what the viewer shows is not what arrived."""
    assert display_image(roi).size != roi.size


def test_a_large_image_is_shown_untouched():
    large = Image.new("RGB", (800, 600))
    assert display_image(large) is large


def test_the_original_is_preferred_over_the_displayed_copy(roi):
    shown = display_image(roi)
    assert image_to_classify(roi, shown) is roi


def test_the_displayed_copy_is_used_when_no_original_was_captured(roi):
    assert image_to_classify(None, roi) is roi


def test_an_empty_viewer_classifies_nothing(roi):
    assert image_to_classify(roi, None) is None


def test_routing_the_original_changes_the_prediction(roi, model_name):
    """The routing is load-bearing: the two images do not score the same."""
    shown = display_image(roi)

    from_original = asyncio.run(app_model.predict_html(image_to_classify(roi, shown), model_name))
    from_upscaled = asyncio.run(app_model.predict_html(shown, model_name))

    assert from_original == asyncio.run(app_model.predict_html(roi, model_name))
    assert from_original != from_upscaled, (
        "upscaling left this image's scores unchanged, so it cannot show what is routed"
    )


def test_gallery_selection_hands_back_the_stored_original(roi):
    """The gallery stores originals, so a click must reach the model unscaled."""
    main = pytest.importorskip("main", reason="needs gradio installed")

    session = None
    try:
        session = main.save_image(roi, None, "D20260107T201908_IFCB134_00042.png")
        event = type("Event", (), {"index": 0})()
        _, _, held = main.on_gallery_select(session, 0, True, event)

        assert held.size == roi.size
    finally:
        if session and "dir" in session:
            shutil.rmtree(session["dir"], ignore_errors=True)
