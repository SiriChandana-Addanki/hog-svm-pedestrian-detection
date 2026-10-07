from scripts.evaluate_video import iou, load_annotations, match_predictions


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
