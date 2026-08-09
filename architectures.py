"""Rebuilding the architecture a checkpoint was trained with.

A checkpoint records its architecture by name (``resnet50``, ``efficientnet_v2_s``
…). :func:`build_model` looks that name up in :data:`MODELS`, constructs the
backbone and swaps its final layer for a head sized to the checkpoint's class
count, so the weights load into the graph they were saved from.

The registry mirrors ``ifcb_classify.models.registry``; adding an architecture
there means adding it here. Unlike the pipeline, backbones are always built with
``weights=None``: every parameter is about to be overwritten by the checkpoint,
so downloading hundreds of megabytes of ImageNet weights on model load would
change nothing but startup time and image size.

That substitution is only sound where the weights argument decides nothing but
the starting values. For a few architectures torchvision also writes its own
constructor arguments when weights are requested, changing the *graph*; those are
enumerated in :func:`_constructor_kwargs`.
"""

import re
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
import torchvision.models as tv


@dataclass(frozen=True)
class ModelSpec:
    """How to build one architecture and where its classification head lives.

    ``head_path`` uses dot notation for attributes and ``[n]`` for Sequential
    indices: ``"fc"`` -> ``model.fc``, ``"classifier[6]"`` -> ``model.classifier[6]``.
    """

    constructor: Any
    head_path: str
    in_features: int
    bias: bool = True


MODELS: dict[str, ModelSpec] = {
    # AlexNet
    "alexnet": ModelSpec(tv.alexnet, "classifier[6]", 4096),
    # ConvNeXt
    "convnext_tiny": ModelSpec(tv.convnext_tiny, "classifier[2]", 768),
    "convnext_small": ModelSpec(tv.convnext_small, "classifier[2]", 768),
    "convnext_base": ModelSpec(tv.convnext_base, "classifier[2]", 1024),
    "convnext_large": ModelSpec(tv.convnext_large, "classifier[2]", 1536),
    # DenseNet
    "densenet121": ModelSpec(tv.densenet121, "classifier", 1024),
    "densenet169": ModelSpec(tv.densenet169, "classifier", 1664),
    "densenet161": ModelSpec(tv.densenet161, "classifier", 2208),
    "densenet201": ModelSpec(tv.densenet201, "classifier", 1920),
    # EfficientNetV2
    "efficientnet_v2_s": ModelSpec(tv.efficientnet_v2_s, "classifier[1]", 1280),
    "efficientnet_v2_m": ModelSpec(tv.efficientnet_v2_m, "classifier[1]", 1280),
    "efficientnet_v2_l": ModelSpec(tv.efficientnet_v2_l, "classifier[1]", 1280),
    # GoogLeNet
    "googlenet": ModelSpec(tv.googlenet, "fc", 1024),
    # Inception
    "inception_v3": ModelSpec(tv.inception_v3, "fc", 2048),
    "inception_v3_untrained": ModelSpec(tv.inception_v3, "fc", 2048),
    # MNASNet
    "mnasnet0_5": ModelSpec(tv.mnasnet0_5, "classifier[1]", 1280),
    "mnasnet0_75": ModelSpec(tv.mnasnet0_75, "classifier[1]", 1280),
    "mnasnet1_0": ModelSpec(tv.mnasnet1_0, "classifier[1]", 1280),
    "mnasnet1_3": ModelSpec(tv.mnasnet1_3, "classifier[1]", 1280),
    # MaxVit
    "maxvit_t": ModelSpec(tv.maxvit_t, "classifier[5]", 512, bias=False),
    # MobileNetV3
    "mobilenet_v3_large": ModelSpec(tv.mobilenet_v3_large, "classifier[3]", 1280),
    "mobilenet_v3_small": ModelSpec(tv.mobilenet_v3_small, "classifier[3]", 1024),
    # ResNet
    "resnet18": ModelSpec(tv.resnet18, "fc", 512),
    "resnet34": ModelSpec(tv.resnet34, "fc", 512),
    "resnet50": ModelSpec(tv.resnet50, "fc", 2048),
    "resnet101": ModelSpec(tv.resnet101, "fc", 2048),
    "resnet152": ModelSpec(tv.resnet152, "fc", 2048),
    # ResNeXt
    "resnext50_32x4d": ModelSpec(tv.resnext50_32x4d, "fc", 2048),
    "resnext101_32x8d": ModelSpec(tv.resnext101_32x8d, "fc", 2048),
    "resnext101_64x4d": ModelSpec(tv.resnext101_64x4d, "fc", 2048),
    # ShuffleNetV2
    "shufflenet_v2_x0_5": ModelSpec(tv.shufflenet_v2_x0_5, "fc", 1024),
    "shufflenet_v2_x1_0": ModelSpec(tv.shufflenet_v2_x1_0, "fc", 1024),
    "shufflenet_v2_x1_5": ModelSpec(tv.shufflenet_v2_x1_5, "fc", 1024),
    "shufflenet_v2_x2_0": ModelSpec(tv.shufflenet_v2_x2_0, "fc", 2048),
    # Swin Transformer V2
    "swin_v2_t": ModelSpec(tv.swin_v2_t, "head", 768),
    "swin_v2_s": ModelSpec(tv.swin_v2_s, "head", 768),
    "swin_v2_b": ModelSpec(tv.swin_v2_b, "head", 1024),
    # Vision Transformer
    "vit_b_16": ModelSpec(tv.vit_b_16, "heads[0]", 768),
    "vit_b_32": ModelSpec(tv.vit_b_32, "heads[0]", 768),
    "vit_l_16": ModelSpec(tv.vit_l_16, "heads[0]", 1024),
    "vit_l_32": ModelSpec(tv.vit_l_32, "heads[0]", 1024),
    "vit_h_14": ModelSpec(tv.vit_h_14, "heads[0]", 1280),
    # VGG
    "vgg11": ModelSpec(tv.vgg11, "classifier[6]", 4096),
    "vgg11_bn": ModelSpec(tv.vgg11_bn, "classifier[6]", 4096),
    "vgg13": ModelSpec(tv.vgg13, "classifier[6]", 4096),
    "vgg13_bn": ModelSpec(tv.vgg13_bn, "classifier[6]", 4096),
    "vgg16": ModelSpec(tv.vgg16, "classifier[6]", 4096),
    "vgg16_bn": ModelSpec(tv.vgg16_bn, "classifier[6]", 4096),
    "vgg19": ModelSpec(tv.vgg19, "classifier[6]", 4096),
    "vgg19_bn": ModelSpec(tv.vgg19_bn, "classifier[6]", 4096),
    # Wide ResNet
    "wide_resnet50_2": ModelSpec(tv.wide_resnet50_2, "fc", 2048),
    "wide_resnet101_2": ModelSpec(tv.wide_resnet101_2, "fc", 2048),
}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build architecture ``name`` with a ``num_classes``-wide classification head.

    ``pretrained`` describes how the checkpoint was *trained*, not what is
    downloaded now — the backbone is always constructed with random weights and
    then overwritten. It still matters, because for a few architectures
    torchvision's pretrained path also changes the graph; see
    :func:`_constructor_kwargs`.

    Raises ``ValueError`` for an unknown name.
    """
    canonical = name if name in MODELS else name.lower()
    spec = MODELS.get(canonical)
    if spec is None:
        raise ValueError(f"Unknown model: {name}. Available: {sorted(MODELS)}")

    model = spec.constructor(**_constructor_kwargs(canonical, spec, pretrained))
    head = nn.Linear(in_features=spec.in_features, out_features=num_classes, bias=spec.bias)
    _set_head(model, spec.head_path, head)
    return model


def _constructor_kwargs(name: str, spec: ModelSpec, pretrained: bool) -> dict:
    """Constructor arguments that reproduce the pipeline's graph without downloading.

    For nearly every architecture the weights argument only decides what the
    parameters start out as, so ``weights=None`` is equivalent. The exceptions are
    the builders that call ``_ovewrite_named_param`` with something other than
    ``num_classes``: they rewrite their own constructor arguments when weights are
    requested, and only reading torchvision's source reveals it.

    * ``inception_v3`` and ``googlenet`` switch on the ImageNet input rescaling,
      and ``googlenet`` additionally drops its auxiliary classifiers after
      loading. Built the plain way, the aux branches make the checkpoint fail to
      load, and the rescaling would have changed every score silently.
    * The Vision Transformer and MaxVit builders size the model from their
      weights' ``min_size``. That is 224 for most of them and so invisible, but
      ``vit_h_14``'s default is the 518-pixel SWAG checkpoint, which changes the
      positional embedding — a pipeline-trained checkpoint would not load into a
      224-pixel model at all. The size is read back from the same metadata
      torchvision reads, so a changed default stays handled.
    """
    kwargs: dict = {"weights": None}

    # inception_v3_untrained is the pipeline's from-scratch spec: it passes
    # weights=None whatever `pretrained` says, so it never takes this path.
    if spec.constructor is tv.inception_v3:
        kwargs["transform_input"] = pretrained and name != "inception_v3_untrained"
        # The pretrained builder skips its own init; so do we. It is slow (a
        # scipy truncated normal per layer) and the checkpoint replaces it all.
        kwargs["init_weights"] = False
    elif spec.constructor is tv.googlenet:
        kwargs["transform_input"] = pretrained
        kwargs["aux_logits"] = not pretrained
        kwargs["init_weights"] = False
    elif pretrained:
        kwargs.update(_pretrained_input_size(name, spec))

    return kwargs


# The keyword each family sizes itself with, and whether it wants a pair.
_INPUT_SIZE_KWARG = {
    "torchvision.models.vision_transformer": ("image_size", False),
    "torchvision.models.maxvit": ("input_size", True),
}


def _pretrained_input_size(name: str, spec: ModelSpec) -> dict:
    """The input-size argument torchvision would derive from the default weights."""
    keyword = _INPUT_SIZE_KWARG.get(spec.constructor.__module__)
    if keyword is None:
        return {}

    min_size = tv.get_model_weights(name).DEFAULT.meta["min_size"]
    argument, wants_pair = keyword
    return {argument: tuple(min_size) if wants_pair else min_size[0]}


def _set_head(model: nn.Module, path: str, layer: nn.Module) -> None:
    """Replace the module at a dot/bracket ``path`` on ``model``."""
    parts = re.split(r"\.", path)
    obj = model
    for part in parts[:-1]:
        obj = _resolve_part(obj, part)
    _assign_part(obj, parts[-1], layer)


def _resolve_part(obj, part: str):
    """Read one path segment off ``obj`` — an attribute, or ``attr[idx]``."""
    match = re.match(r"(\w+)\[(\d+)]", part)
    if match:
        attr, idx = match.group(1), int(match.group(2))
        return getattr(obj, attr)[idx]
    return getattr(obj, part)


def _assign_part(obj, part: str, value: nn.Module) -> None:
    """Assign ``value`` to one path segment of ``obj`` — an attribute, or ``attr[idx]``."""
    match = re.match(r"(\w+)\[(\d+)]", part)
    if match:
        attr, idx = match.group(1), int(match.group(2))
        getattr(obj, attr)[idx] = value
    else:
        setattr(obj, part, value)
