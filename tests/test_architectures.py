"""Rebuilding a backbone must reproduce the pipeline's graph, not just its weights.

Backbones are built with ``weights=None`` here, since the checkpoint overwrites
every parameter and downloading ImageNet weights would only cost time and image
size. That is safe exactly as long as the weights argument decides nothing but
the starting values. For the GoogLeNet family it decides more: asking torchvision
for pretrained weights also switches on the ImageNet input rescaling, and for
``googlenet`` drops the auxiliary classifiers. These tests pin that difference so
it cannot be optimised away.
"""

import inspect
import re
import sys

import pytest
import torch.nn as nn
import torchvision.models as tv

from architectures import MODELS, _constructor_kwargs, build_model


def test_head_is_resized_to_the_checkpoint_class_count():
    model = build_model("resnet50", 17)

    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 17


def test_nested_head_paths_are_resolved():
    model = build_model("efficientnet_v2_s", 5)

    assert model.classifier[1].out_features == 5


def test_unknown_architecture_is_refused():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("resnet51", 5)


def test_architecture_names_are_matched_case_insensitively():
    assert build_model("ResNet50", 5).fc.out_features == 5


@pytest.mark.parametrize(
    ("name", "pretrained", "transform_input", "aux_logits"),
    [
        # Pretrained googlenet/inception_v3 rescale their input; from-scratch ones do not.
        ("googlenet", True, True, False),
        ("googlenet", False, False, True),
        ("inception_v3", True, True, True),
        ("inception_v3", False, False, True),
        # The pipeline's from-scratch spec passes weights=None whatever it is told.
        ("inception_v3_untrained", True, False, True),
        ("inception_v3_untrained", False, False, True),
    ],
)
def test_googlenet_family_graph_matches_how_it_was_trained(
    name, pretrained, transform_input, aux_logits
):
    model = build_model(name, 5, pretrained)

    assert model.transform_input is transform_input
    assert model.aux_logits is aux_logits


def test_googlenet_drops_its_auxiliary_classifiers_when_pretrained():
    """torchvision removes them after loading, so a checkpoint has no keys for them."""
    assert build_model("googlenet", 5, pretrained=True).aux1 is None
    assert build_model("googlenet", 5, pretrained=False).aux1 is not None


def test_no_architecture_downloads_pretrained_weights(monkeypatch):
    """A container must not fetch hundreds of megabytes it is about to overwrite."""
    import torch.hub

    def fail(*args, **kwargs):
        raise AssertionError("build_model attempted to download pretrained weights")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fail)

    for name in ("resnet50", "googlenet", "inception_v3", "densenet121"):
        build_model(name, 5, pretrained=True)


@pytest.mark.parametrize(
    "name", ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14", "maxvit_t"]
)
def test_transformers_are_sized_the_way_their_weights_would_size_them(name):
    """torchvision sizes these from ``weights.meta["min_size"]``, not from a default.

    Mostly 224 and invisible — but ``vit_h_14``'s default is the 518-pixel SWAG
    checkpoint, and a model built at 224 has a differently shaped positional
    embedding, so a pipeline-trained checkpoint would not load into it at all.
    Read from torchvision's own metadata so it cannot drift into a stale literal.
    """
    expected = tv.get_model_weights(name).DEFAULT.meta["min_size"]
    kwargs = _constructor_kwargs(name, MODELS[name], pretrained=True)

    assert kwargs.get("image_size", kwargs.get("input_size")) in (expected[0], tuple(expected))
    assert "image_size" not in _constructor_kwargs(name, MODELS[name], pretrained=False)


def test_vit_h_14_is_the_one_that_actually_differs():
    """Guards the premise above: if this stops being true, the test lost its teeth."""
    assert tv.get_model_weights("vit_h_14").DEFAULT.meta["min_size"][0] == 518
    assert tv.get_model_weights("vit_b_16").DEFAULT.meta["min_size"][0] == 224


# Every constructor argument torchvision rewrites when pretrained weights are
# requested, and how each is accounted for. A new entry here means a builder
# started changing its graph based on the weights argument, which is exactly the
# substitution architectures.build_model relies on being safe.
KNOWN_REWRITES = {
    # Handled explicitly in _constructor_kwargs.
    ("googlenet", "aux_logits"), ("googlenet", "init_weights"), ("googlenet", "transform_input"),
    ("inception", "aux_logits"), ("inception", "init_weights"), ("inception", "transform_input"),
    ("maxvit", "input_size"), ("vision_transformer", "image_size"),
    # Rewritten unconditionally, outside the `if weights is not None` branch, so
    # both sides get them either way.
    ("resnet", "groups"), ("resnet", "width_per_group"),
}


def test_no_new_torchvision_builder_rewrites_its_own_arguments():
    found = set()
    for spec in MODELS.values():
        source = inspect.getsource(sys.modules[spec.constructor.__module__])
        module = spec.constructor.__module__.rsplit(".", 1)[-1]
        for param in re.findall(r'_ovewrite_named_param\(kwargs,\s*"(\w+)"', source):
            if param != "num_classes":
                found.add((module, param))

    assert found == KNOWN_REWRITES, (
        f"torchvision's parameter rewrites changed: {found ^ KNOWN_REWRITES}. "
        "Check whether architectures._constructor_kwargs still reproduces the graph."
    )


@pytest.mark.parametrize(
    "name",
    ["alexnet", "convnext_tiny", "densenet121", "efficientnet_v2_s", "googlenet",
     "mnasnet1_0", "mobilenet_v3_large", "resnet18", "resnext50_32x4d",
     "shufflenet_v2_x1_0", "swin_v2_t", "vit_b_16", "vgg11", "wide_resnet50_2"],
)
def test_one_architecture_per_family_gets_its_head_attached(name):
    """A wrong head path or feature count only shows up when the model is built.

    One entry per family, since a family shares its head path — building all
    fifty would spend most of a minute and gigabytes on the largest variants
    without testing anything the small ones do not. Cross-family drift from the
    pipeline's registry is caught in ``test_pipeline_equivalence.py``.
    """
    model = build_model(name, 3, pretrained=False)
    heads = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 3]

    assert heads, f"{name}: no 3-class head was attached"
