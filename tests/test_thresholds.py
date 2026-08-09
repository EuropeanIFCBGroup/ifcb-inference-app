"""Finding a model's thresholds, and turning scores into the class the pipeline records."""

import json

import numpy as np
import pytest

from model import (
    UNCLASSIFIED,
    find_thresholds_file,
    load_thresholds,
    resolve_class_names,
    thresholds_fitted_on_augmented,
)

LABELS = ["Alpha_spp", "Beta_spp", "Gamma_spp"]


def _write(path, metrics, **meta):
    path.write_text(json.dumps({"class_metrics": metrics, **meta}))
    return str(path)


def test_a_plain_thresholds_json_is_found(tmp_path):
    (tmp_path / "thresholds.json").write_text("{}")

    assert find_thresholds_file(str(tmp_path)) == str(tmp_path / "thresholds.json")


def test_the_name_training_writes_is_found(tmp_path):
    """A directory copied straight out of a training run has no thresholds.json."""
    (tmp_path / "V7-resnet50_b64_thresholds_and_metrics.json").write_text("{}")

    assert find_thresholds_file(str(tmp_path)).endswith("_thresholds_and_metrics.json")


def test_a_plain_thresholds_json_wins_over_a_run_named_one(tmp_path):
    (tmp_path / "thresholds.json").write_text("{}")
    (tmp_path / "V7_thresholds_and_metrics.json").write_text("{}")

    assert find_thresholds_file(str(tmp_path)).endswith("thresholds.json")
    assert not find_thresholds_file(str(tmp_path)).endswith("_and_metrics.json")


def test_ambiguous_thresholds_files_are_not_guessed_between(tmp_path):
    (tmp_path / "V6_thresholds_and_metrics.json").write_text("{}")
    (tmp_path / "V7_thresholds_and_metrics.json").write_text("{}")

    assert find_thresholds_file(str(tmp_path)) is None


def test_a_directory_with_no_thresholds_yields_none(tmp_path):
    assert find_thresholds_file(str(tmp_path)) is None


def test_thresholds_keyed_by_class_name_are_read(tmp_path):
    path = _write(tmp_path / "t.json", {"Beta_spp": {"threshold": 0.4, "f1": 0.9, "support": 12}})

    thresholds, _, details = load_thresholds(path, LABELS)

    assert thresholds == {"Beta_spp": 0.4}
    assert details["Beta_spp"] == {"f1": 0.9, "support": 12}


def test_thresholds_keyed_by_class_index_are_resolved(tmp_path):
    """Older files number their classes instead of naming them."""
    path = _write(tmp_path / "t.json", {"2": {"threshold": 0.7}})

    thresholds, _, _ = load_thresholds(path, LABELS)

    assert thresholds == {"Gamma_spp": 0.7}


def test_a_missing_thresholds_file_is_not_an_error():
    assert load_thresholds(None, LABELS) == ({}, {}, {})


def test_a_file_recording_its_validation_transform_is_current():
    assert not thresholds_fitted_on_augmented({"validation_transform": "dataset_squarepad"})


@pytest.mark.parametrize("meta", [{}, {"validation_transform": None}, {"model_name": "V7"}])
def test_a_file_without_a_validation_transform_predates_the_fix(meta):
    assert thresholds_fitted_on_augmented(meta)


def test_a_score_over_its_threshold_keeps_the_class():
    scores = np.array([0.1, 0.8, 0.1])

    assert resolve_class_names(scores, LABELS, {"Beta_spp": 0.4}) == ("Beta_spp", "Beta_spp")


def test_a_score_under_its_threshold_is_unclassified():
    scores = np.array([0.1, 0.3, 0.1])

    assert resolve_class_names(scores, LABELS, {"Beta_spp": 0.4}) == ("Beta_spp", UNCLASSIFIED)


def test_a_score_exactly_on_its_threshold_is_accepted():
    """The pipeline compares with >=, so the boundary belongs to the class."""
    scores = np.array([0.1, 0.4, 0.1])

    assert resolve_class_names(scores, LABELS, {"Beta_spp": 0.4})[1] == "Beta_spp"


def test_a_class_with_no_threshold_is_accepted():
    scores = np.array([0.1, 0.2, 0.9])

    assert resolve_class_names(scores, LABELS, {"Beta_spp": 0.4})[1] == "Gamma_spp"


def test_a_nan_threshold_means_no_threshold():
    """The pipeline fills NaN for classes a file omits and reads it as 'accept'."""
    scores = np.array([0.1, 0.2, 0.9])

    assert resolve_class_names(scores, LABELS, {"Gamma_spp": float("nan")})[1] == "Gamma_spp"
