"""Loading models and scoring images the way ifcb-classify's ``infer`` does.

Everything that decides a score is read from the checkpoint rather than assumed:
the architecture, the input size, the transform and the dataset statistics all
come out of the training config the checkpoint carries. A model trained with a
different backbone, a different input size or a normalised transform therefore
scores here exactly as it does in the pipeline, without this app being changed.

The transform is resolved through :func:`preprocessing.eval_transform_name`, so
the training augmentation never runs while scoring — see that module for why.
"""

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch

from architectures import build_model
from checkpoints import load_checkpoint
from preprocessing import build_eval_transform, eval_transform_name

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

NUM_TOP_CLASSES = 5

MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')

UNCLASSIFIED = "unclassified"

# Key recording the transform a thresholds file's validation split was read
# through. Only releases that stopped augmenting that split write it, so its
# absence dates the file: those thresholds were fitted against randomly jittered
# and flipped images and no longer match the model's operating point.
VALIDATION_TRANSFORM_KEY = "validation_transform"

# Assumed validation fraction for checkpoints that record no ``val_split``,
# used only to turn a class's validation support into a training-set estimate
# for the About table.
DEFAULT_VAL_SPLIT = 0.2


@dataclass(frozen=True)
class LoadedModel:
    """A model ready to score with, plus everything the UI reports about it."""

    labels: tuple[str, ...]
    net: torch.nn.Module
    transform: object
    thresholds: dict[str, float]
    threshold_meta: dict
    class_details: dict[str, dict]
    num_params: int
    config: dict
    transform_name: str


def _find_weights_file(model_dir):
    for ext in ('weights.pth', 'weights.pt'):
        path = os.path.join(model_dir, ext)
        if os.path.isfile(path):
            return path
    return None


def discover_models():
    models_found = {}
    for name in sorted(os.listdir(MODELS_DIR), reverse=True):
        model_dir = os.path.join(MODELS_DIR, name)
        if not os.path.isdir(model_dir):
            continue
        weights_path = _find_weights_file(model_dir)
        if weights_path is None:
            continue
        has_classes_txt = os.path.isfile(os.path.join(model_dir, 'classes.txt'))
        if has_classes_txt or _weights_have_class_names(weights_path):
            models_found[name.replace('-', ' ')] = name
    return models_found


def _weights_have_class_names(weights_path):
    try:
        data = torch.load(weights_path, map_location='cpu', weights_only=True)
    except Exception:
        try:
            data = torch.load(weights_path, map_location='cpu', weights_only=False)
        except Exception:
            return False
    return isinstance(data, dict) and 'class_names' in data


AVAILABLE_MODELS = discover_models()
DEFAULT_MODEL = next(iter(AVAILABLE_MODELS)) if AVAILABLE_MODELS else None

_loaded_models = {}


def format_label(name):
    return name.replace('_', ' ')


def find_thresholds_file(model_dir):
    """Locate a model directory's thresholds JSON, or ``None`` if it has none.

    Mirrors the pipeline's resolution order. Training writes
    ``{run_name}_thresholds_and_metrics.json``, so a directory packaged straight
    out of a training run carries that name rather than ``thresholds.json``;
    looking only for the latter would silently apply no thresholds at all and
    quietly disagree with the pipeline on exactly the ROIs thresholds exist to
    catch. An ambiguous match resolves to nothing rather than guessing.
    """
    plain = os.path.join(model_dir, 'thresholds.json')
    if os.path.isfile(plain):
        return plain

    candidates = sorted(
        os.path.join(model_dir, name) for name in os.listdir(model_dir)
        if name.endswith("_thresholds_and_metrics.json")
    )
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        log.warning(
            "%s holds %d thresholds files and no thresholds.json; not guessing between "
            "them, so no thresholds will be applied. Leave exactly one, or rename the "
            "intended one to thresholds.json.",
            model_dir, len(candidates),
        )
    return None


def load_thresholds(path, labels):
    """Read a thresholds JSON into ``(thresholds, meta, class_details)``.

    Keys are class names; older files key by the class's integer index instead,
    which is resolved against ``labels``. Classes the file does not mention get
    no threshold, which means their predictions are always accepted.
    """
    if not path or not os.path.exists(path):
        return {}, {}, {}
    with open(path) as f:
        data = json.load(f)
    thresholds = {}
    class_details = {}
    for key, metrics in data.get("class_metrics", {}).items():
        if key.isdigit():
            idx = int(key)
            label = labels[idx] if idx < len(labels) else None
        else:
            label = key
        if label is None:
            continue
        thresholds[label] = metrics["threshold"]
        class_details[label] = {
            "f1": metrics.get("f1"),
            "support": metrics.get("support"),
        }
    meta = {k: v for k, v in data.items() if k != "class_metrics"}
    return thresholds, meta, class_details


def thresholds_fitted_on_augmented(threshold_meta):
    """Whether a thresholds file was fitted against an augmented validation split.

    True when it records no ``validation_transform``. Files written before that
    split stopped being augmented lack the key, as does any hand-written one.

    Callers must first establish that there *is* a thresholds file: a model with
    none has nothing that could have been fitted wrongly, and its empty metadata
    is indistinguishable here from a real file that records nothing.
    """
    return threshold_meta.get(VALIDATION_TRANSFORM_KEY) is None


def get_model(name=None):
    """Load (and cache) a model by its display name."""
    if name is None:
        name = DEFAULT_MODEL
    if name in _loaded_models:
        return _loaded_models[name]

    dir_name = AVAILABLE_MODELS[name]
    model_dir = os.path.join(MODELS_DIR, dir_name)
    weights_path = _find_weights_file(model_dir)

    checkpoint = load_checkpoint(weights_path, os.path.join(model_dir, 'classes.txt'))
    config = checkpoint["config"]
    labels = checkpoint["class_names"]

    net = build_model(config["model"], len(labels), config.get("pretrained", True))
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    net.to(DEVICE)

    transform_name = eval_transform_name(config["transform"])
    if transform_name != config["transform"]:
        log.info(
            "%s was trained with '%s'; scoring with '%s' (augmentation is training-only).",
            name, config["transform"], transform_name,
        )
    transform = build_eval_transform(
        transform_name,
        config.get("image_width", 224),
        config.get("image_height", 224),
        config.get("mean"),
        config.get("std"),
    )

    thresholds, threshold_meta, class_details = load_thresholds(
        find_thresholds_file(model_dir), labels
    )
    if thresholds and thresholds_fitted_on_augmented(threshold_meta):
        log.warning(
            "%s: thresholds.json records no '%s', so it predates the de-augmented "
            "validation split and its thresholds were fitted on randomly jittered and "
            "flipped images. They no longer match this model's operating point.",
            name, VALIDATION_TRANSFORM_KEY,
        )

    _loaded_models[name] = LoadedModel(
        labels=tuple(labels),
        net=net,
        transform=transform,
        thresholds=thresholds,
        threshold_meta=threshold_meta,
        class_details=class_details,
        num_params=sum(p.numel() for p in net.parameters()),
        config=config,
        transform_name=transform_name,
    )
    return _loaded_models[name]


def resolve_class_names(scores, labels, thresholds):
    """Resolve one ROI's scores into ``(class_name_auto, class_name)``.

    ``class_name_auto`` is the raw argmax label; ``class_name`` applies the
    argmax class's threshold and falls back to ``"unclassified"`` when the score
    does not reach it.

    A class with no threshold is always accepted. So is one whose threshold is
    NaN: the pipeline fills NaN for classes a thresholds file omits and reads it
    as "no threshold", where a plain comparison would instead reject everything.

    This is the per-image form of the pipeline's array-wide resolution, so the
    label shown here is the one its class-scores output would record.
    """
    best = int(np.argmax(scores))
    class_name_auto = labels[best]
    threshold = thresholds.get(class_name_auto)
    if threshold is None or np.isnan(threshold) or scores[best] >= threshold:
        return class_name_auto, class_name_auto
    return class_name_auto, UNCLASSIFIED


def score_image(image, model_name=None):
    """Score one PIL image, returning ``(LoadedModel, scores)`` aligned to its labels."""
    loaded = get_model(model_name)
    tensor = loaded.transform(image.convert('RGB')).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = loaded.net(tensor)[0]
        probs = torch.nn.functional.softmax(logits, dim=0)

    return loaded, probs.cpu().numpy()


def render_predictions(predictions, model_name=None, resolved=None):
    if not predictions:
        return '<div class="pred-panel"><p class="pred-empty">No predictions</p></div>'

    class_thresholds = get_model(model_name).thresholds if model_name else {}

    sorted_preds = sorted(
        predictions.items(), key=lambda x: x[1], reverse=True
    )[:NUM_TOP_CLASSES]

    model_subtitle = ""
    if model_name:
        model_subtitle = (
            f'<div class="pred-model-name"'
            f' style="font-size:0.85em;opacity:0.6;margin-bottom:8px">'
            f'{model_name}</div>'
        )

    resolved_html = ""
    if resolved is not None:
        unclassified = resolved == UNCLASSIFIED
        resolved_html = (
            f'<div class="pred-resolved{" pred-resolved-none" if unclassified else ""}">'
            f'<span class="pred-resolved-label">Classified as</span>'
            f'<span class="pred-resolved-value">{resolved}</span>'
            f'</div>'
        )

    rows = []
    for name, prob in sorted_preds:
        threshold = class_thresholds.get(name)
        pct = prob * 100

        threshold_html = ""
        if threshold is not None:
            t_pct = threshold * 100
            threshold_html = (
                f'<div class="pred-threshold" '
                f'style="left:{t_pct:.1f}%" '
                f'title="Threshold: {threshold:.2f}"></div>'
            )

        rows.append(
            f'<div class="pred-row">'
            f'<div class="pred-header">'
            f'<span class="pred-name">{name}</span>'
            f'<span class="pred-pct">{pct:.1f}%</span>'
            f'</div>'
            f'<div class="pred-track">'
            f'<div class="pred-fill" style="width:{pct:.1f}%"></div>'
            f'{threshold_html}'
            f'</div>'
            f'</div>'
        )

    legend = (
        '<div class="pred-legend">'
        '<span class="pred-legend-marker"></span>'
        '<span>F1 threshold</span>'
        '</div>'
    )

    return (
        '<div class="pred-panel">' + model_subtitle + resolved_html
        + ''.join(rows) + legend + '</div>'
    )


async def predict(image, model_name=None):
    if image is None:
        return {}
    loaded, scores = score_image(image, model_name=model_name)
    return {label: float(scores[i]) for i, label in enumerate(loaded.labels)}


async def predict_scores(image, model_name=None):
    """Return every class score, plus the class the pipeline would record.

    Returns:
        ``{"class_labels": [...], "scores": [...], "class_name_auto": str,
        "class_name": str}`` — ``scores`` ordered to match ``class_labels``,
        ``class_name_auto`` the argmax label and ``class_name`` the same label
        after its threshold is applied, or ``"unclassified"``.
    """
    if image is None:
        return {"class_labels": [], "scores": [], "class_name_auto": None, "class_name": None}

    loaded, scores = score_image(image, model_name=model_name)
    class_name_auto, class_name = resolve_class_names(scores, loaded.labels, loaded.thresholds)
    return {
        "class_labels": list(loaded.labels),
        "scores": [float(s) for s in scores],
        "class_name_auto": class_name_auto,
        "class_name": class_name,
    }


async def predict_html(image, model_name=None):
    if image is None:
        return render_predictions({})
    loaded, scores = score_image(image, model_name=model_name)
    _, class_name = resolve_class_names(scores, loaded.labels, loaded.thresholds)
    predictions = {label: float(scores[i]) for i, label in enumerate(loaded.labels)}
    return render_predictions(predictions, model_name=model_name, resolved=class_name)


def get_thresholds(model_name=None):
    """Return per-class thresholds and class labels for a model.

    Returns a JSON-serialisable dict with:
      - class_labels: list of class names (same order as score columns)
      - thresholds: dict mapping class name -> optimal F1 threshold
      - validation_transform: the transform the thresholds were fitted through,
        or null for a file written before that was recorded
      - model_name: display name passed in (or the default)
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    loaded = get_model(model_name)
    return {
        "class_labels": list(loaded.labels),
        "thresholds": loaded.thresholds,
        "validation_transform": loaded.threshold_meta.get(VALIDATION_TRANSFORM_KEY),
        "model_name": model_name,
    }


def build_about_markdown(model_name=None):
    loaded = get_model(model_name)
    config = loaded.config

    val_split = config.get("val_split") or DEFAULT_VAL_SPLIT
    train_per_val = (1 - val_split) / val_split

    header = (
        "| # | Class | F1 | Train | Val |\n"
        "|---|-------|---:|------:|----:|\n"
    )
    rows = []
    for i, name in enumerate(loaded.labels, 1):
        details = loaded.class_details.get(name, {})
        f1 = details.get("f1")
        support = details.get("support")
        f1_str = f"{f1:.2f}" if f1 is not None else "-"
        train_str = str(round(support * train_per_val)) if support is not None else "-"
        val_str = str(support) if support is not None else "-"
        rows.append(
            f"| {i} | {format_label(name)} | {f1_str} | {train_str} | {val_str} |"
        )
    class_list = header + "\n".join(rows)

    meta_model_name = loaded.threshold_meta.get("model_name", "")
    model_name_line = f"- **Model name:** {meta_model_name}\n" if meta_model_name else ""

    trained_with = config["transform"]
    transform_line = f"- **Transform:** `{loaded.transform_name}`"
    if trained_with != loaded.transform_name:
        transform_line += f" (trained with `{trained_with}`; augmentation is training-only)"

    padding = {
        "dataset_squarepad": "square-padded",
        "dataset_fullpad": "padded to size",
        "dataset_reflectpad": "reflection-padded",
    }.get(loaded.transform_name.replace("_normalised", ""), "not padded")

    warning = ""
    if loaded.thresholds and thresholds_fitted_on_augmented(loaded.threshold_meta):
        warning = (
            "\n> **Note:** this model's thresholds record no `validation_transform`, so they "
            "were fitted against a validation split that still had the training augmentation "
            "applied. The scores above are unaffected, but the thresholds no longer match the "
            "model's operating point and are due to be refitted.\n"
        )

    dir_name = AVAILABLE_MODELS[model_name or DEFAULT_MODEL]
    about_path = os.path.join(MODELS_DIR, dir_name, "about.md")
    about_section = ""
    if os.path.isfile(about_path):
        with open(about_path) as f:
            about_section = f.read().strip() + "\n\n"

    return (
        "## Model\n\n"
        f"{model_name_line}"
        f"- **Architecture:** {config['model']}\n"
        f"- **Parameters:** {loaded.num_params:,}\n"
        f"- **Input size:** {config.get('image_width', 224)} x "
        f"{config.get('image_height', 224)} ({padding})\n"
        f"{transform_line}\n"
        f"- **Classes:** {len(loaded.labels)}\n"
        f"- **Device:** {DEVICE}\n"
        f"{warning}\n"
        f"{about_section}"
        "## Class list\n\n"
        f"{class_list}\n"
    )
