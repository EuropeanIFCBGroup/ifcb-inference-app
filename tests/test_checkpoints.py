"""Reading checkpoints that describe themselves, and ones that do not."""

import pytest
import torch

from checkpoints import LEGACY_CONFIG, guess_model_name, load_checkpoint, read_class_names


def _resnet50_keys():
    return {"conv1.weight": torch.zeros(1), "layer4.0.conv1.weight": torch.zeros(1),
            "fc.weight": torch.zeros(1)}


def _efficientnet_keys():
    return {"features.0.0.weight": torch.zeros(1), "classifier.1.weight": torch.zeros(1)}


def test_pipeline_checkpoint_keeps_its_own_config(tmp_path):
    path = tmp_path / "weights.pt"
    config = {"model": "efficientnet_v2_s", "transform": "dataset_fullpad_augmented_normalised",
              "image_width": 300, "image_height": 300, "mean": 0.5, "std": 0.2}
    torch.save({"state_dict": _resnet50_keys(), "class_names": ["a", "b"], "config": config}, path)

    loaded = load_checkpoint(str(path))

    assert loaded["config"] == config
    assert loaded["class_names"] == ["a", "b"]


def test_legacy_checkpoint_gets_the_pipeline_defaults(tmp_path):
    path = tmp_path / "weights.pth"
    torch.save(_resnet50_keys(), path)
    (tmp_path / "classes.txt").write_text("Alpha_spp\nBeta_spp\n")

    loaded = load_checkpoint(str(path), str(tmp_path / "classes.txt"))

    assert loaded["config"] == {**LEGACY_CONFIG, "model": "resnet50"}
    assert loaded["class_names"] == ["Alpha_spp", "Beta_spp"]


def test_legacy_architecture_is_inferred_not_assumed(tmp_path):
    """Assuming resnet50 would make an EfficientNet checkpoint fail to load."""
    path = tmp_path / "weights.pth"
    torch.save(_efficientnet_keys(), path)
    (tmp_path / "classes.txt").write_text("Alpha_spp\n")

    assert load_checkpoint(str(path), str(tmp_path / "classes.txt"))["config"]["model"] == (
        "efficientnet_v2_s"
    )


def test_unrecognised_layout_defaults_to_resnet50():
    assert guess_model_name({"something.weight": torch.zeros(1)}) == "resnet50"


def test_missing_class_list_is_reported(tmp_path):
    path = tmp_path / "weights.pth"
    torch.save(_resnet50_keys(), path)

    with pytest.raises(FileNotFoundError, match="class list"):
        load_checkpoint(str(path), str(tmp_path / "classes.txt"))


def test_class_names_keep_quotes_so_threshold_lookups_still_match(tmp_path):
    """Thresholds are keyed by class name, so the label text must match the pipeline's."""
    path = tmp_path / "classes.txt"
    path.write_text("'Alpha_spp'\nBeta_spp\n")

    assert read_class_names(str(path)) == ["'Alpha_spp'", "Beta_spp"]


def test_windows_line_endings_are_stripped(tmp_path):
    path = tmp_path / "classes.txt"
    path.write_bytes(b"Alpha_spp\r\nBeta_spp\r\n")

    assert read_class_names(str(path)) == ["Alpha_spp", "Beta_spp"]


def test_corrupt_checkpoint_names_the_file(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"not a checkpoint at all")

    with pytest.raises(RuntimeError, match="weights.pt"):
        load_checkpoint(str(path))
