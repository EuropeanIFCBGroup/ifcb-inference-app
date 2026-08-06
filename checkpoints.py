"""Reading model checkpoints into a single shape the rest of the app can use.

Two formats are accepted, mirroring ``ifcb_classify.checkpoint.load_checkpoint``:

* **Pipeline checkpoints** — a dict carrying ``state_dict``, ``class_names`` and
  the training ``config``. The config is what makes a checkpoint self-describing:
  architecture, input size, transform and dataset statistics all come from it, so
  the app scores a model the way it was trained rather than the way it was
  hardcoded.
* **Legacy checkpoints** — a bare state-dict saved outside the pipeline, with
  class names in a sibling ``classes.txt``. These record nothing about how they
  were trained, so they get the same defaults the pipeline assumes for them.

Both are normalised to ``{state_dict, class_names, config}``.
"""

import logging
import os
import pickle

import torch

log = logging.getLogger(__name__)

# What the pipeline assumes for a checkpoint that describes nothing about itself.
# Changing these silently rescores every legacy model, so they are pinned here
# rather than spread through the loading code. The architecture is not among them
# — it is inferred from the weights by :func:`guess_model_name`.
LEGACY_CONFIG = {
    "image_width": 224,
    "image_height": 224,
    "transform": "dataset_squarepad",
}


def load_checkpoint(weights_path: str, classes_path: str | None = None) -> dict:
    """Load ``weights_path`` into a ``{state_dict, class_names, config}`` dict.

    ``classes_path`` supplies the class list for legacy checkpoints; pipeline
    checkpoints carry their own and ignore it.

    Raises ``FileNotFoundError`` when a legacy checkpoint has no class list, and
    ``RuntimeError`` when the file cannot be read at all.
    """
    data = _torch_load(weights_path)

    if isinstance(data, dict) and "state_dict" in data and "config" in data:
        return {
            "state_dict": data["state_dict"],
            "class_names": list(data["class_names"]),
            "config": dict(data["config"]),
        }

    # A dict with class names but no config: an intermediate export format.
    if isinstance(data, dict) and "state_dict" in data:
        state_dict = data["state_dict"]
        return {
            "state_dict": state_dict,
            "class_names": list(data.get("class_names") or read_class_names(classes_path)),
            "config": _legacy_config(state_dict),
        }

    return {
        "state_dict": data,
        "class_names": read_class_names(classes_path),
        "config": _legacy_config(data),
    }


def _legacy_config(state_dict) -> dict:
    """The training config assumed for a checkpoint that records none."""
    return {**LEGACY_CONFIG, "model": guess_model_name(state_dict)}


def guess_model_name(state_dict) -> str:
    """Infer the architecture of a bare state-dict from its parameter names.

    Recognises the two layouts the pipeline recognises — ResNet-50 and
    EfficientNetV2-S — and defaults to ``resnet50``. A wrong guess fails loudly
    when the weights are loaded rather than scoring anything incorrectly.
    """
    keys = set(state_dict.keys())
    if any(k.startswith("layer4") for k in keys) and "fc.weight" in keys:
        return "resnet50"
    if any(k.startswith("features") for k in keys) and "classifier.1.weight" in keys:
        return "efficientnet_v2_s"
    return "resnet50"


def read_class_names(classes_path: str | None) -> list[str]:
    """Read class names, one per line, from a ``classes.txt``.

    Only surrounding whitespace is stripped, matching the pipeline exactly. An
    earlier version of this app also stripped quotes, which is worse than it
    sounds: thresholds are keyed by class name, so a quoted list would leave the
    app looking up names the thresholds file does not contain while the pipeline
    matched them.
    """
    if not classes_path or not os.path.isfile(classes_path):
        raise FileNotFoundError(
            f"No class list found at {classes_path!r}. A checkpoint that does not embed "
            "its class names needs a classes.txt beside it."
        )
    with open(classes_path) as f:
        return [line.strip() for line in f if line.strip()]


def _torch_load(weights_path: str):
    """Load a checkpoint, preferring torch's safe mode.

    ``weights_only=True`` refuses to unpickle arbitrary Python objects. Pipeline
    checkpoints load fine under it; older ones holding pickled objects do not, and
    fall back to a full unpickle. That fallback can execute code from the file, so
    it is logged.

    The pipeline gates the same fallback behind an explicit ``--allow-unsafe``
    because it loads whatever path a user names on the command line. This app has
    no such input: it only ever loads what an operator placed in ``data/models/``,
    which in a container means at image-build time. The gate is therefore dropped
    on purpose rather than overlooked — but it does mean a model directory is as
    trusted as the code, so populate it accordingly.
    """
    try:
        return torch.load(weights_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        log.warning(
            "%s could not be loaded in torch's safe mode, which usually means it is an "
            "older checkpoint holding pickled Python objects. Falling back to a full "
            "unpickle — only load model files from trusted sources.",
            weights_path,
        )
    except RuntimeError as err:
        raise RuntimeError(
            f"Could not read checkpoint {weights_path}: {err}. It may be truncated or corrupt."
        ) from err

    try:
        return torch.load(weights_path, map_location="cpu", weights_only=False)
    except Exception as err:
        raise RuntimeError(
            f"Could not read checkpoint {weights_path} even with a full unpickle, so it is "
            "most likely corrupt or truncated."
        ) from err
