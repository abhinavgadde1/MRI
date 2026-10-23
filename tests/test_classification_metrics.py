"""F1 ≡ binary Dice from a confusion matrix."""

from segmentation.classification_metrics import confusion_counts, scores_from_counts
import numpy as np


def test_f1_equals_dice_and_standard_ratios():
    pred = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    gt = np.array([1, 0, 0, 1, 1, 0], dtype=bool)
    cm = confusion_counts(pred, gt)
    assert cm == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    s = scores_from_counts(**cm)
    assert s["precision"] == 2 / 3
    assert s["recall"] == 2 / 3
    assert s["specificity"] == 2 / 3
    assert s["accuracy"] == 4 / 6
    expected = 2 * 2 / (2 * 2 + 1 + 1)
    assert abs(s["f1"] - expected) < 1e-12
    assert abs(s["dice"] - expected) < 1e-12
    assert abs(s["f1"] - s["dice"]) < 1e-12


def test_empty_prediction_precision_nan_recall_zero():
    pred = np.zeros(4, dtype=bool)
    gt = np.array([1, 0, 1, 0], dtype=bool)
    s = scores_from_counts(**confusion_counts(pred, gt))
    assert s["precision"] != s["precision"]  # nan
    assert s["recall"] == 0.0
    assert s["dice"] == 0.0
    assert s["f1"] == 0.0
