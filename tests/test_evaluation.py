import warnings

import pytest

from scripts.evaluate_video import (
    iou,
    load_annotations,
    match_predictions,
    match_predictions_with_details,
)
from src.annotation_validation import warn_if_extreme_annotation


def test_csv_xywh_maps_to_xyxy(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text("x,y,w,h\n10,20,30,40\n", encoding="utf-8")

    assert load_annotations(path) == {1: [[10, 20, 40, 60]]}


def test_iou_and_one_to_one_prediction_matching():
    truth = [0, 0, 10, 10]
    assert iou(truth, [0, 0, 10, 10]) == 1.0
    assert iou(truth, [20, 20, 30, 30]) == 0.0

    tp, fp, fn = match_predictions(
        [
            {"bbox": [0, 0, 10, 10], "confidence": 0.9},
            {"bbox": [0, 0, 10, 10], "confidence": 0.8},
        ],
        [truth],
    )
    assert (tp, fp, fn) == (1, 1, 0)


def test_match_predictions_converts_xywh_to_xyxy():
    tp, fp, fn = match_predictions(
        [{"bbox": [10, 20, 30, 40], "confidence": 0.9}],
        [[10, 20, 40, 60]],
    )

    assert (tp, fp, fn) == (1, 0, 0)


def test_detailed_matching_preserves_confidence_order_and_reports_each_box():
    detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.9},
        {"bbox": [0, 0, 10, 10], "confidence": 0.8},
    ]
    truth = [[0, 0, 10, 10], [30, 30, 40, 40]]

    details = match_predictions_with_details(detections, truth)

    assert details["matches"] == [{
        "prediction": detections[0],
        "ground_truth_index": 0,
        "iou": 1.0,
    }]
    assert details["false_positives"] == [detections[1]]
    assert details["false_negatives"] == [{
        "ground_truth_index": 1,
        "bbox": truth[1],
    }]
    assert match_predictions(detections, truth) == (1, 1, 1)


def test_extreme_annotation_warns_and_is_not_modified():
    box = [631, -375, 3142, 4647]

    with pytest.warns(UserWarning, match="annotation row/frame 1.*extreme size") as caught:
        warn_if_extreme_annotation(
            1,
            box,
            (1080, 1920, 3),
            source="dataset/fourway.csv",
        )

    assert "coordinates extend outside the frame" in str(caught[0].message)
    assert "608.1% of frame area" in str(caught[0].message)
    assert "retained unchanged" in str(caught[0].message)
    assert box == [631, -375, 3142, 4647]


def test_normal_annotation_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_extreme_annotation(
            1,
            [10, 20, 40, 80],
            (1080, 1920, 3),
        )

    assert not caught
