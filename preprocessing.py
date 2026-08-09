"""Image preprocessing, kept equivalent to the ifcb-classify inference path.

A checkpoint records the *name* of the transform it was trained with. That name
bundles two things which behave differently once training is over:
preprocessing (padding, resize, normalisation), which must match what the model
saw, and augmentation (random flips, brightness/contrast jitter), which must not
run at all. Scoring through an augmented pipeline makes every result a single
random draw, so the same image classifies differently depending on the RNG
position.

:func:`build_eval_transform` is the only entry point, and it resolves the name
through :func:`eval_transform_name` before dispatching. The augmented pipelines
are therefore not implemented here at all: this app only ever scores images, so
a code path that could only ever be wrong is better absent than available.

This is a deliberate copy of ``ifcb_classify.data.transforms`` and the
non-augmented half of ``ifcb_classify.data.datasets.build_transform``. Depending
on the package instead would pull scikit-learn, OpenCV, h5py, pandas, SciPy,
matplotlib, plotly and a git-sourced ifcbkit into a four-package deployment.
Changes to the upstream transforms must be mirrored here — ``tests/`` asserts
the two agree tensor-for-tensor when the pipeline is importable.
"""

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import v2 as transforms


class SquarePad:
    """Pad an image to square using the mean of its four corner pixels."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        s = image.size()
        max_wh = np.max([s[-1], s[-2]])
        hp = int((max_wh - s[-1]) / 2)
        vp = int((max_wh - s[-2]) / 2)

        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, _corner_mean(image), "constant")


class FullPad:
    """Centre-pad an image up to a target size, or pass it through if already larger."""

    def __init__(self, target_width: int, target_height: int):
        self.target_width = target_width
        self.target_height = target_height

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        s = image.size()
        width = s[-1]
        height = s[-2]

        if width > self.target_width and height > self.target_height:
            return image

        horizontal_offset = int((self.target_width - width) / 2)
        vertical_offset = int((self.target_height - height) / 2)

        padding = (horizontal_offset, vertical_offset)
        return F.pad(image, padding, _corner_mean(image), "constant")


class ReflectPad(torch.nn.Module):
    """Reflection-pad an image toward a target size, then resize to exactly it."""

    def __init__(self, target_width: int = 299, target_height: int = 299):
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # OpenCV is not a dependency of this app, and no released model uses this
        # transform. Reimplementing BORDER_REFLECT and INTER_NEAREST on top of
        # torchvision differs from cv2 in edge and half-pixel handling, so the
        # scores would disagree with the pipeline silently. Failing loudly is the
        # honest option; install opencv-python to use such a checkpoint.
        try:
            import cv2
        except ImportError as err:
            raise ImportError(
                "The 'dataset_reflectpad' transform needs OpenCV, which this app does not "
                "install by default. Run `pip install opencv-python-headless` to score a "
                "checkpoint trained with it."
            ) from err

        s = image.size()
        width = s[-1]
        height = s[-2]

        if width > self.target_width and height > self.target_height:
            return image

        horizontal_offset = max(int((self.target_width - width) / 2), 0)
        vertical_offset = max(int((self.target_height - height) / 2), 0)

        numpy_image = image.numpy()
        cv2_image = np.transpose(numpy_image, (1, 2, 0))
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        cv2_image = cv2.copyMakeBorder(
            cv2_image, vertical_offset, vertical_offset,
            horizontal_offset, horizontal_offset, cv2.BORDER_REFLECT,
        )
        cv2_image = cv2.resize(
            cv2_image, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST
        )
        new_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        new_image = np.transpose(new_image, (2, 0, 1))
        return torch.from_numpy(new_image)


def _corner_mean(image: torch.Tensor) -> float:
    """Estimate the background colour as the mean of the four corner pixels."""
    top_left = image[0][0][0]
    bottom_left = image[0][-1][0]
    top_right = image[0][0][-1]
    bottom_right = image[0][-1][-1]
    return (
        top_left.item() + bottom_left.item() + top_right.item() + bottom_right.item()
    ) / 4


# Every transform name a checkpoint may record, mapped to the one used for
# scoring. Augmented names lose only their random operations; the padding,
# resize and normalisation that the model was trained on are kept.
_EVAL_TRANSFORM_NAMES = {
    "dataset_squarepad_augmented": "dataset_squarepad",
    "dataset_fullpad_augmented": "dataset_fullpad",
    "dataset_squarepad_augmented_normalised": "dataset_squarepad_normalised",
    "dataset_fullpad_augmented_normalised": "dataset_fullpad_normalised",
}

EVAL_TRANSFORM_NAMES = (
    "dataset",
    "dataset_normalised",
    "dataset_squarepad",
    "dataset_squarepad_normalised",
    "dataset_fullpad",
    "dataset_fullpad_normalised",
    "dataset_reflectpad",
)


def eval_transform_name(name: str) -> str:
    """Return the augmentation-free counterpart of transform ``name``.

    Names that carry no augmentation pass through unchanged.
    """
    return _EVAL_TRANSFORM_NAMES.get(name, name)


def build_eval_transform(
    name: str,
    width: int = 224,
    height: int = 224,
    mean: float | None = None,
    std: float | None = None,
) -> transforms.Compose:
    """Build the scoring pipeline for the transform a checkpoint was trained with.

    ``name`` is taken straight from the checkpoint and resolved through
    :func:`eval_transform_name`, so augmented names are accepted and their random
    operations dropped.

    All variants convert to 3-channel grayscale float tensors first: IFCB images
    are single-channel but the backbones expect RGB. ``_normalised`` variants
    require ``mean`` and ``std`` — the single-channel dataset statistics recorded
    in the training config, replicated across the three channels.

    Raises ``ValueError`` for an unknown name, or for a ``_normalised`` name with
    no statistics to normalise by.
    """
    resolved = eval_transform_name(name)

    base = [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
    resize = transforms.Resize((width, height), antialias=True)

    if resolved not in EVAL_TRANSFORM_NAMES:
        raise ValueError(
            f"Unknown transform: {name!r}. This app scores with one of "
            f"{list(EVAL_TRANSFORM_NAMES)} (augmented names are accepted and de-augmented)."
        )

    if resolved.startswith("dataset_squarepad"):
        pad = [SquarePad()]
    elif resolved.startswith("dataset_fullpad"):
        pad = [FullPad(width, height)]
    elif resolved == "dataset_reflectpad":
        pad = [ReflectPad(width, height)]
    else:
        pad = []

    steps = [*base, *pad, resize]

    if resolved.endswith("_normalised"):
        if mean is None or std is None:
            raise ValueError(
                f"Transform {resolved!r} requires the dataset mean and std, which this "
                "checkpoint does not record. It cannot be scored faithfully without them."
            )
        steps.append(transforms.Normalize(mean=[mean] * 3, std=[std] * 3))

    return transforms.Compose(steps)
