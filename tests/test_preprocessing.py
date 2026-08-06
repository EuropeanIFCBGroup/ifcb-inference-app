"""The scoring transform must depend only on the image and on the checkpoint."""

import numpy as np
import pytest
import torch
from PIL import Image

from preprocessing import (
    EVAL_TRANSFORM_NAMES,
    FullPad,
    SquarePad,
    build_eval_transform,
    eval_transform_name,
)


@pytest.fixture
def roi():
    """A small non-square grayscale image, like an IFCB ROI."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (37, 61), dtype=np.uint8)).convert("RGB")


@pytest.mark.parametrize(
    ("trained_with", "expected"),
    [
        ("dataset_squarepad_augmented", "dataset_squarepad"),
        ("dataset_fullpad_augmented", "dataset_fullpad"),
        ("dataset_squarepad_augmented_normalised", "dataset_squarepad_normalised"),
        ("dataset_fullpad_augmented_normalised", "dataset_fullpad_normalised"),
        ("dataset_squarepad", "dataset_squarepad"),
        ("dataset", "dataset"),
    ],
)
def test_augmentation_is_stripped_but_preprocessing_is_kept(trained_with, expected):
    assert eval_transform_name(trained_with) == expected


@pytest.mark.parametrize("name", EVAL_TRANSFORM_NAMES)
def test_scoring_is_deterministic(name, roi):
    """No random operation may survive into the scoring pipeline.

    Two passes of one transform over one image must be identical. An augmented
    pipeline would fail this: its RNG advances between the calls.
    """
    if name == "dataset_reflectpad":
        pytest.importorskip("cv2")
    stats = {"mean": 0.69, "std": 0.13} if name.endswith("_normalised") else {}
    transform = build_eval_transform(name, 224, 224, **stats)

    torch.testing.assert_close(transform(roi), transform(roi), rtol=0, atol=0)


def test_augmented_name_scores_like_its_de_augmented_counterpart(roi):
    """Passing the checkpoint's own name in must not reintroduce augmentation."""
    trained = build_eval_transform("dataset_squarepad_augmented", 224, 224)
    plain = build_eval_transform("dataset_squarepad", 224, 224)

    torch.testing.assert_close(trained(roi), plain(roi), rtol=0, atol=0)


def test_stats_alone_do_not_trigger_normalisation(roi):
    """A checkpoint can record mean/std under a non-normalised transform name.

    The shipped V7 model does exactly that. Normalising on the presence of the
    statistics rather than on the transform name would silently whiten its input.
    """
    without = build_eval_transform("dataset_squarepad", 224, 224)
    with_unused_stats = build_eval_transform("dataset_squarepad", 224, 224, mean=0.69, std=0.13)

    torch.testing.assert_close(without(roi), with_unused_stats(roi), rtol=0, atol=0)


def test_normalised_transform_applies_the_recorded_stats(roi):
    plain = build_eval_transform("dataset_squarepad", 224, 224)(roi)
    normalised = build_eval_transform(
        "dataset_squarepad_normalised", 224, 224, mean=0.69, std=0.13
    )(roi)

    torch.testing.assert_close(normalised, (plain - 0.69) / 0.13, rtol=1e-6, atol=1e-6)


def test_normalised_transform_without_stats_is_refused():
    with pytest.raises(ValueError, match="mean and std"):
        build_eval_transform("dataset_squarepad_normalised", 224, 224)


def test_unknown_transform_is_refused():
    with pytest.raises(ValueError, match="Unknown transform"):
        build_eval_transform("dataset_rotated", 224, 224)


def test_transform_honours_the_checkpoint_input_size(roi):
    assert build_eval_transform("dataset_squarepad", 299, 299)(roi).shape == (3, 299, 299)


def test_square_pad_pads_the_short_side_to_the_long_one():
    image = torch.zeros(3, 10, 30)
    assert SquarePad()(image).shape == (3, 30, 30)


def test_square_pad_fills_with_the_corner_background():
    image = torch.full((3, 10, 30), 0.25)
    padded = SquarePad()(image)

    assert padded[0, 0, 0].item() == pytest.approx(0.25)


def test_full_pad_leaves_a_larger_image_alone():
    image = torch.zeros(3, 300, 300)
    assert FullPad(224, 224)(image).shape == (3, 300, 300)
