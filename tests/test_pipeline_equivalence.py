"""The app must score an image exactly as ``ifcb-classify infer`` scores it.

These tests run the *real* pipeline against the app on real IFCB ROIs, so they
need a checkout of ifcb-classify and a directory of raw bins. Point
``IFCB_CLASSIFY_SRC`` and ``IFCB_DATA_DIR`` at them; without both, everything
here skips, so the suite still runs on a machine that only has the app.

Two cases matter:

* the shipped checkpoint, which is the regression guard — its scores must not
  move; and
* the same checkpoint with a ``_normalised`` transform, which no shipped model
  uses yet but the retrained one may. Rewriting the config exercises that branch
  today rather than discovering it diverged after the new weights land.
"""

import os
import shutil
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

PIPELINE_SRC = os.environ.get("IFCB_CLASSIFY_SRC", "/media/anders/work/R/ifcb-pytorch-classify/src")
DATA_DIR = os.environ.get("IFCB_DATA_DIR", "/media/anders/work/ifcb/data/data")
N_ROIS = 40

if os.path.isdir(PIPELINE_SRC) and PIPELINE_SRC not in sys.path:
    sys.path.insert(0, PIPELINE_SRC)

pytest.importorskip("ifcbkit", reason="raw-bin reading needs ifcbkit")
datasets = pytest.importorskip(
    "ifcb_classify.data.datasets", reason=f"no ifcb-classify checkout at {PIPELINE_SRC}"
)
factory = pytest.importorskip("ifcb_classify.models.factory")

import model as app_model  # noqa: E402
from checkpoints import load_checkpoint  # noqa: E402
from preprocessing import build_eval_transform, eval_transform_name  # noqa: E402


def _first_bin():
    for root, _, files in os.walk(DATA_DIR):
        for name in sorted(files):
            if name.endswith(".roi"):
                return os.path.join(root, name)
    return None


@pytest.fixture(scope="module")
def rois():
    """The first ROIs of a real bin, decoded the way the pipeline decodes them."""
    from ifcb_classify.data.ifcb_bin import iter_bin_images

    bin_path = _first_bin()
    if bin_path is None:
        pytest.skip(f"no raw IFCB bins under {DATA_DIR}")

    images = []
    for _, image in iter_bin_images(bin_path):
        images.append(image)
        if len(images) == N_ROIS:
            break
    return images


def _installed_checkpoints():
    for display_name, dir_name in app_model.AVAILABLE_MODELS.items():
        model_dir = os.path.join(app_model.MODELS_DIR, dir_name)
        path = app_model._find_weights_file(model_dir)
        yield display_name, path, load_checkpoint(path, os.path.join(model_dir, "classes.txt"))


@pytest.fixture(scope="module")
def pipeline_checkpoint():
    """The shipped checkpoint that carries a full training config."""
    for entry in _installed_checkpoints():
        if "val_split" in entry[2]["config"]:
            return entry
    pytest.skip("no checkpoint with an embedded training config is installed")


@pytest.fixture(scope="module")
def legacy_checkpoint():
    """A shipped bare state-dict, which describes nothing about how it was trained."""
    for entry in _installed_checkpoints():
        if "val_split" not in entry[2]["config"]:
            return entry
    pytest.skip("no legacy checkpoint is installed")


def pipeline_scores(checkpoint, images):
    """Score images the way ``ifcb_classify.infer`` does, using the pipeline's own code."""
    config = checkpoint["config"]
    class_names = checkpoint["class_names"]

    transform = datasets.build_transform(
        datasets.eval_transform_name(config["transform"]),
        config["image_width"],
        config["image_height"],
        config.get("mean"),
        config.get("std"),
    )
    net = factory.get_model(config["model"], len(class_names), config.get("pretrained", True))
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()

    with torch.no_grad():
        batch = torch.stack([transform(image) for image in images])
        return torch.nn.functional.softmax(net(batch), dim=1).numpy()


def app_scores(display_name, images):
    """Score images through the app's own loading and prediction path."""
    return np.array([
        app_model.score_image(image, model_name=display_name)[1] for image in images
    ])


def test_architecture_registry_matches_the_pipeline():
    """The registry is a copy, so it can drift. This is what notices.

    Only the fields that decide the graph are compared: the pipeline additionally
    records which pretrained weights to download, which this app never does.
    """
    from ifcb_classify.models.registry import MODELS as pipeline_models

    from architectures import MODELS as app_models

    assert set(app_models) == set(pipeline_models) - {"custom"}, "registry names have drifted"

    for name, expected in pipeline_models.items():
        if name == "custom":
            continue
        actual = app_models[name]
        assert actual.constructor is expected.constructor, f"{name}: different constructor"
        assert actual.head_path == expected.head_path, f"{name}: different head path"
        assert actual.in_features == expected.in_features, f"{name}: different head width"
        assert actual.bias == expected.bias, f"{name}: different head bias"

    # The app has no ``weights`` field: it never downloads. It instead hardcodes
    # that ``inception_v3_untrained`` is the one spec built from scratch however
    # ``pretrained`` is set. If upstream adds another, that shortcut goes stale.
    from_scratch = {n for n, s in pipeline_models.items() if s.weights is None}
    assert from_scratch == {"inception_v3_untrained"}, (
        f"the pipeline now builds {from_scratch} from scratch; "
        "architectures._constructor_kwargs only knows about inception_v3_untrained"
    )


@pytest.mark.parametrize(
    "name",
    ["resnet50", "googlenet", "inception_v3", "inception_v3_untrained", "maxvit_t",
     "vit_b_16", "resnext50_32x4d", "efficientnet_v2_s"],
)
@pytest.mark.parametrize("pretrained", [True, False])
def test_rebuilt_graph_matches_the_pipeline(name, pretrained, monkeypatch):
    """Same parameter names and shapes as the pipeline builds, so checkpoints load.

    The pipeline asks torchvision for pretrained weights; this app never does.
    Comparing the two means building the pipeline's side once, with the download
    and the load stubbed out — the shapes are decided by the constructor, not by
    the bytes. The largest ViTs are left out: their families are represented by
    ``vit_b_16``, and ``test_architectures.py`` covers what differs between them
    without allocating a 630-million-parameter model twice.
    """
    from torchvision.models._api import WeightsEnum

    from architectures import build_model

    monkeypatch.setattr(WeightsEnum, "get_state_dict", lambda self, *a, **k: {})
    monkeypatch.setattr(nn.Module, "load_state_dict", lambda self, sd, *a, **k: None)

    def shapes(module):
        return {key: tuple(value.shape) for key, value in module.state_dict().items()}

    expected = shapes(factory.get_model(name, 7, pretrained))
    actual = shapes(build_model(name, 7, pretrained))

    assert actual == expected


@pytest.mark.parametrize("trained_with", datasets.TRANSFORM_NAMES)
def test_transform_names_resolve_the_same_way(trained_with):
    """Every name the pipeline accepts must de-augment to the same thing here."""
    assert eval_transform_name(trained_with) == datasets.eval_transform_name(trained_with)


@pytest.mark.parametrize("trained_with", datasets.TRANSFORM_NAMES)
def test_every_transform_produces_the_pipeline_s_tensor(trained_with, rois):
    """The vendored transforms must be equivalent, not merely believed equivalent.

    Tested against the pipeline's own ``build_transform`` for every name it
    accepts, including the branches no installed model uses — otherwise the only
    thing checking this copy is the copy itself.
    """
    name = datasets.eval_transform_name(trained_with)
    if name == "dataset_reflectpad":
        pytest.importorskip("cv2")

    stats = {"mean": 0.6901, "std": 0.1278}
    expected = datasets.build_transform(name, 224, 224, **stats)
    actual = build_eval_transform(trained_with, 224, 224, **stats)

    for roi in rois[:5]:
        torch.testing.assert_close(actual(roi), expected(roi), rtol=0, atol=0)


def test_non_square_input_sizes_agree(rois):
    """Width and height are passed through in the same order as the pipeline."""
    expected = datasets.build_transform("dataset_fullpad", 200, 260)
    actual = build_eval_transform("dataset_fullpad_augmented", 200, 260)

    torch.testing.assert_close(actual(rois[0]), expected(rois[0]), rtol=0, atol=0)


def test_shipped_model_scores_identically(pipeline_checkpoint, rois):
    """The regression guard: the installed model's scores must not move."""
    display_name, _, checkpoint = pipeline_checkpoint

    expected = pipeline_scores(checkpoint, rois)
    actual = app_scores(display_name, rois)

    assert (expected.argmax(1) == actual.argmax(1)).all(), "top class differs from the pipeline"
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def test_legacy_checkpoint_falls_back_to_the_pipeline_defaults(legacy_checkpoint):
    """A bare state-dict must be scored on the same assumptions the pipeline makes."""
    from ifcb_classify.checkpoint import load_checkpoint as pipeline_load

    display_name, weights_path, checkpoint = legacy_checkpoint
    model_dir = os.path.dirname(weights_path)
    expected = pipeline_load(
        weights_path, classes_path=os.path.join(model_dir, "classes.txt"), allow_unsafe=True
    )

    assert checkpoint["config"] == expected["config"]
    assert list(checkpoint["class_names"]) == list(expected["class_names"])


def test_legacy_checkpoint_scores_identically(legacy_checkpoint, rois):
    display_name, _, checkpoint = legacy_checkpoint

    expected = pipeline_scores(checkpoint, rois)
    actual = app_scores(display_name, rois)

    assert (expected.argmax(1) == actual.argmax(1)).all(), "top class differs from the pipeline"
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def test_normalised_checkpoint_scores_identically(pipeline_checkpoint, rois, tmp_path, monkeypatch):
    """The forward guard: a ``_normalised`` checkpoint must agree too.

    No shipped model is trained this way, so the config is rewritten to make one.
    The weights no longer suit the inputs and the predictions are meaningless —
    the point is only that both sides produce the *same* meaningless numbers, via
    a normalisation branch nothing else exercises.
    """
    _, weights_path, checkpoint = pipeline_checkpoint
    config = {**checkpoint["config"], "transform": "dataset_squarepad_augmented_normalised"}
    assert config.get("mean") and config.get("std"), "checkpoint records no dataset statistics"

    model_dir = tmp_path / "Rewritten-Normalised-Model"
    model_dir.mkdir()
    torch.save(
        {**checkpoint, "config": config},
        model_dir / os.path.basename(weights_path),
    )
    thresholds = os.path.join(os.path.dirname(weights_path), "thresholds.json")
    if os.path.isfile(thresholds):
        shutil.copy(thresholds, model_dir / "thresholds.json")

    display_name = _register_temporary_model(model_dir, monkeypatch)
    expected = pipeline_scores({**checkpoint, "config": config}, rois)
    actual = app_scores(display_name, rois)

    transform = build_eval_transform(
        eval_transform_name(config["transform"]), config["image_width"],
        config["image_height"], config["mean"], config["std"],
    )
    plain = build_eval_transform("dataset_squarepad", config["image_width"], config["image_height"])
    assert not torch.equal(transform(rois[0]), plain(rois[0])), "normalisation was not applied"

    assert (expected.argmax(1) == actual.argmax(1)).all(), "top class differs from the pipeline"
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def _register_temporary_model(model_dir, monkeypatch):
    """Make a model directory outside ``data/models`` visible to the app's loader.

    Every global it touches is replaced through ``monkeypatch``, so pytest
    restores them however the test ends. Restoring by hand would leave
    ``MODELS_DIR`` pointing at a temporary directory for the rest of the module
    if the test failed partway, and the module-scoped fixtures share it.
    """
    display_name = model_dir.name.replace("-", " ")
    monkeypatch.setitem(app_model.AVAILABLE_MODELS, display_name, model_dir.name)
    monkeypatch.setattr(app_model, "MODELS_DIR", str(model_dir.parent))
    monkeypatch.setattr(app_model, "_loaded_models", dict(app_model._loaded_models))
    return display_name
