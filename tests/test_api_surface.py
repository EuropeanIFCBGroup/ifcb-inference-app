"""The published API is a contract with tools outside this repository.

iRfcb's ``ifcb_classify_images()`` posts an image and a model name to
``/predict_html`` and reads ``/get_thresholds``. Gradio names an endpoint after
its handler whenever ``api_name`` is not given, so a purely internal rewiring can
rename or reshape a published endpoint without anything in this repo changing:
routing the gallery-click handler through a ``gr.State`` once dropped the image
argument from ``/predict_html`` entirely, because Gradio omits State components
from the API.

These tests read main.py's source, so they need no gradio installed.
"""

import ast
import os

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The endpoints external tools call. Adding one is a deliberate act; losing or
# renaming one breaks a downstream caller that this repo cannot see.
PUBLISHED_ENDPOINTS = {"predict_html", "predict_scores", "get_thresholds"}

# Components Gradio excludes from the API surface. An endpoint that takes one
# has that argument silently missing from its published signature.
API_INVISIBLE_COMPONENTS = {"State"}


def _listeners():
    """Yield every event-listener call in main.py as (method, keywords)."""
    tree = ast.parse(open(os.path.join(APP_DIR, "main.py")).read())
    events = {"click", "upload", "select", "change", "clear", "then", "submit"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in events:
            continue
        yield node.func.attr, {kw.arg: kw.value for kw in node.keywords}


def _api_name(keywords):
    """The literal api_name of a listener: a string, False, or None if unset."""
    node = keywords.get("api_name")
    return node.value if isinstance(node, ast.Constant) else None


def test_published_endpoints_are_exactly_the_declared_contract():
    declared = {
        name for _, kw in _listeners()
        if isinstance(name := _api_name(kw), str)
    }

    assert declared == PUBLISHED_ENDPOINTS, (
        "the published API changed. Renaming or removing an endpoint breaks callers "
        "outside this repository, such as iRfcb::ifcb_classify_images()."
    )


def test_no_listener_publishes_an_endpoint_by_accident():
    """Anything without an explicit api_name gets one generated from its handler."""
    unmarked = [
        method for method, kw in _listeners()
        if "api_name" not in kw
    ]

    assert not unmarked, (
        f"{len(unmarked)} listener(s) ({sorted(set(unmarked))}) declare no api_name, so "
        "gradio will publish them under their handler's name and the API surface will "
        "track internal rewiring. Pass api_name=False for UI-only listeners."
    )


@pytest.mark.parametrize("endpoint", sorted(PUBLISHED_ENDPOINTS))
def test_published_endpoints_take_no_state_argument(endpoint):
    """A gr.State input is dropped from the signature callers actually see."""
    tree = ast.parse(open(os.path.join(APP_DIR, "main.py")).read())
    state_variables = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", None) in API_INVISIBLE_COMPONENTS
    }

    for _, keywords in _listeners():
        if _api_name(keywords) != endpoint:
            continue
        inputs = keywords.get("inputs")
        names = [e.id for e in inputs.elts if isinstance(e, ast.Name)] if inputs else []
        offending = [n for n in names if n in state_variables]

        assert not offending, (
            f"/{endpoint} takes {offending}, which gradio omits from the API — "
            "external callers would be sending an argument the endpoint no longer declares."
        )
        return

    pytest.fail(f"no listener declares api_name={endpoint!r}")


def test_predict_html_still_accepts_an_image():
    """iRfcb posts [image, model_name]; the endpoint must accept both, in order."""
    for _, keywords in _listeners():
        if _api_name(keywords) != "predict_html":
            continue
        inputs = keywords["inputs"]
        names = [e.id for e in inputs.elts if isinstance(e, ast.Name)]

        assert len(names) == 2, f"/predict_html takes {names}, expected an image and a model"
        assert "image" in names[0], f"first argument is {names[0]!r}, expected an image component"
        assert "model" in names[1], f"second argument is {names[1]!r}, expected a model dropdown"
        return

    pytest.fail("no listener declares api_name='predict_html'")
