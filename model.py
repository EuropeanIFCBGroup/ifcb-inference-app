import json
import os

import torch
import torchvision.models as models
from torchvision.transforms import v2 as transforms

from utils.CustomTransforms import SquarePad

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

NUM_TOP_CLASSES = 5

MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')


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
        has_embedded_classes = _weights_have_class_names(weights_path)
        if has_classes_txt or has_embedded_classes:
            display_name = name.replace('-', ' ')
            models_found[display_name] = name
    return models_found


def _weights_have_class_names(weights_path):
    try:
        data = torch.load(weights_path, map_location='cpu', weights_only=False)
        return isinstance(data, dict) and 'class_names' in data
    except Exception:
        return False


AVAILABLE_MODELS = discover_models()
DEFAULT_MODEL = next(iter(AVAILABLE_MODELS)) if AVAILABLE_MODELS else None

_loaded_models = {}


def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip().strip("'").strip() for line in f if line.strip()]


def _load_weights(weights_path):
    data = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    if isinstance(data, dict) and 'state_dict' in data:
        return data['state_dict'], data.get('class_names')
    return data, None


def build_resnet50(num_classes, weights_path):
    net = models.resnet50()
    net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    state_dict, _ = _load_weights(weights_path)
    net.load_state_dict(state_dict)
    net.eval()
    return net.to(DEVICE)


image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    SquarePad(),
    transforms.Resize((224, 224), antialias=True),
])


def format_label(name):
    return name.replace('_', ' ')


def load_thresholds(path, labels):
    if not os.path.exists(path):
        return {}, {}, {}
    with open(path, 'r') as f:
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
    meta = {
        k: v for k, v in data.items()
        if k != "class_metrics"
    }
    return thresholds, meta, class_details


def get_model(name=None):
    if name is None:
        name = DEFAULT_MODEL
    if name in _loaded_models:
        return _loaded_models[name]

    dir_name = AVAILABLE_MODELS[name]
    model_dir = os.path.join(BASE_DIR, 'data', 'models', dir_name)

    weights_path = _find_weights_file(model_dir)
    classes_path = os.path.join(model_dir, 'classes.txt')

    if os.path.isfile(classes_path):
        labels = load_labels(classes_path)
    else:
        _, embedded_names = _load_weights(weights_path)
        labels = list(embedded_names)

    net = build_resnet50(len(labels), weights_path)
    thresholds, threshold_meta, class_details = load_thresholds(
        os.path.join(model_dir, 'thresholds.json'), labels
    )
    num_params = sum(p.numel() for p in net.parameters())

    _loaded_models[name] = (
        labels, net, thresholds, threshold_meta, num_params, class_details
    )
    return _loaded_models[name]


def render_predictions(predictions, model_name=None):
    if not predictions:
        return '<div class="pred-panel"><p class="pred-empty">No predictions</p></div>'

    if model_name:
        _, _, class_thresholds, _, _, _ = get_model(model_name)
    else:
        class_thresholds = {}

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
        '<span>F2 threshold</span>'
        '</div>'
    )

    return '<div class="pred-panel">' + model_subtitle + ''.join(rows) + legend + '</div>'


async def predict(image, model_name=None):
    if image is None:
        return {}

    labels, net, _, _, _, _ = get_model(model_name)

    image_rgb = image.convert('RGB')
    tensor = image_transform(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = net(tensor)[0]
        probs = torch.nn.functional.softmax(logits, dim=0)

    return {
        labels[i]: float(probs[i])
        for i in range(len(labels))
    }


async def predict_scores(image, model_name=None):
    """Return all class scores as a JSON-serialisable dict.

    Returns:
        {"class_labels": [...], "scores": [...]}
    where scores are ordered to match class_labels.
    """
    preds = await predict(image, model_name=model_name)
    if not preds:
        return {"class_labels": [], "scores": []}
    labels, _, _, _, _, _ = get_model(model_name)
    ordered_labels = list(labels)
    ordered_scores = [preds.get(label, 0.0) for label in ordered_labels]
    return {"class_labels": ordered_labels, "scores": ordered_scores}


async def predict_html(image, model_name=None):
    preds = await predict(image, model_name=model_name)
    return render_predictions(preds, model_name=model_name)


def get_thresholds(model_name=None):
    """Return per-class thresholds and class labels for a model.

    Returns a JSON-serialisable dict with:
      - class_labels: list of class names (same order as score columns)
      - thresholds: dict mapping class name -> optimal F2 threshold
      - model_name: display name passed in (or the default)
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    labels, _, thresholds, _, _, _ = get_model(model_name)
    return {
        "class_labels": list(labels),
        "thresholds": thresholds,
        "model_name": model_name,
    }


def build_about_markdown(model_name=None):
    labels, _, _, threshold_meta, num_params, class_details = get_model(
        model_name
    )

    header = (
        "| # | Class | F1 | Train | Test |\n"
        "|---|-------|---:|------:|-----:|\n"
    )
    rows = []
    for i, name in enumerate(labels, 1):
        details = class_details.get(name, {})
        f1 = details.get("f1")
        support = details.get("support")
        f1_str = f"{f1:.2f}" if f1 is not None else "-"
        if support is not None:
            train_str = str(support * 4)
            test_str = str(support)
        else:
            train_str = "-"
            test_str = "-"
        rows.append(
            f"| {i} | {format_label(name)} "
            f"| {f1_str} | {train_str} | {test_str} |"
        )
    class_list = header + "\n".join(rows)
    meta_model_name = threshold_meta.get("model_name", "")
    model_name_line = f"- **Model name:** {meta_model_name}\n" if meta_model_name else ""

    dir_name = AVAILABLE_MODELS[model_name or DEFAULT_MODEL]
    about_path = os.path.join(MODELS_DIR, dir_name, "about.md")
    about_section = ""
    if os.path.isfile(about_path):
        with open(about_path, "r") as f:
            about_section = f.read().strip() + "\n\n"

    return (
        "## Model\n\n"
        f"{model_name_line}"
        f"- **Architecture:** ResNet-50\n"
        f"- **Parameters:** {num_params:,}\n"
        f"- **Input size:** 224 x 224 (square-padded)\n"
        f"- **Classes:** {len(labels)}\n"
        f"- **Device:** {DEVICE}\n\n"
        f"{about_section}"
        "## Class list\n\n"
        f"{class_list}\n"
    )
