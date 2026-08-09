"""The prediction panel's custom CSS has to survive both colour themes.

Gradio exposes two kinds of colour variable, and they behave differently:
``--neutral-50`` … ``--neutral-950`` are fixed palette entries with the same
value in light and dark, while the semantic ones (``--body-text-color``,
``--background-fill-secondary``) are redefined per theme.

Painting a background with a fixed palette entry is therefore a trap. In dark
mode Gradio sets ``--body-text-color`` to ``--neutral-100``, so a
``--neutral-100`` background renders its own text invisible — which is exactly
what the resolved-class box did when it was first added.
"""

import ast
import os
import re

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The palette entries Gradio reuses as text colours, so a background must not be
# one of them: --neutral-100 is the dark-mode body text, --neutral-800 the light.
TEXT_COLOURED_PALETTE_VARS = ("--neutral-100", "--neutral-800", "--neutral-900", "--neutral-50")


def _css():
    """The `css` string literal from main.py, read without importing gradio."""
    tree = ast.parse(open(os.path.join(APP_DIR, "main.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "css":
            return node.value.value
    raise AssertionError("main.py no longer defines a module-level `css` string")


def _rules():
    """Yield (selector, body) for each rule in the stylesheet, comments removed."""
    stylesheet = re.sub(r"/\*.*?\*/", "", _css(), flags=re.DOTALL)
    for block in stylesheet.split("}"):
        if "{" not in block:
            continue
        selector, body = block.split("{", 1)
        yield selector.strip(), body


def test_no_text_bearing_box_paints_itself_the_dark_mode_text_colour():
    offenders = []
    for selector, body in _rules():
        declarations = [d.strip() for d in body.split(";") if d.strip()]
        background = next((d for d in declarations if d.startswith("background")), None)
        if background and any(var in background for var in TEXT_COLOURED_PALETTE_VARS):
            offenders.append(f"{selector} ({background})")

    assert not offenders, (
        "these rules paint a background with a fixed palette variable that Gradio also "
        f"uses as a text colour, so their text can vanish in one theme: {offenders}"
    )


def test_the_resolved_class_box_uses_theme_aware_colours():
    body = dict(_rules())[".pred-resolved"]

    assert "--background-fill-secondary" in body, "background must flip with the theme"
    assert "--body-text-color" in body, "text colour must flip with the theme"
