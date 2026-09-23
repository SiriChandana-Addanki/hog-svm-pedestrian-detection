from src.postprocessing import non_max_suppression


def test_nms_suppresses_overlapping_boxes():
    detections = [
        {
            "bbox": [10, 10, 100, 100],
            "confidence": 0.95,
        },
        {
            "bbox": [20, 20, 100, 100],
            "confidence": 0.80,
        },
        {
            "bbox": [300, 300, 50, 50],
            "confidence": 0.70,
        },
    ]

    result = non_max_suppression(
        detections,
        iou_threshold=0.3,
    )

    assert len(result) == 2

    assert result[0]["confidence"] == 0.95
    assert result[1]["confidence"] == 0.70