import numpy as np
from unittest.mock import Mock, patch

from src.config import DetectorConfig
from src.detector import HOGPedestrianDetector


def test_detector_initializes():
    detector = HOGPedestrianDetector()

    assert detector.hog is not None
    assert detector.hog.getDescriptorSize() == 3780
    assert detector.config.hit_threshold == 1.0


def test_detector_returns_expected_structure():
    detector = HOGPedestrianDetector()

    image = np.zeros(
        (128, 64, 3),
        dtype=np.uint8,
    )

    result = detector.detect(image)

    assert "detections" in result
    assert "count" in result
    assert "latency_ms" in result

    assert isinstance(result["detections"], list)
    assert isinstance(result["count"], int)
    assert isinstance(result["latency_ms"], float)
    assert isinstance(result["cpu_time_ms"], float)


def test_detector_preserves_original_boxes_and_confidence_and_passes_threshold():
    hog = Mock()
    hog.detectMultiScale.return_value = (
        [(5, 7, 64, 128)],
        np.array([[1.25]], dtype=np.float32),
    )
    config = DetectorConfig(hit_threshold=1.0)
    image = np.zeros((200, 100, 3), dtype=np.uint8)

    with patch("src.detector.cv2.HOGDescriptor", return_value=hog), patch(
        "src.detector.cv2.HOGDescriptor_getDefaultPeopleDetector",
        return_value=object(),
    ):
        detector = HOGPedestrianDetector(config)
        result = detector.detect(image)

    assert result["detections"] == [
        {"bbox": [5, 7, 64, 128], "confidence": 1.25}
    ]
    assert result["count"] == 1
    call = hog.detectMultiScale.call_args
    assert call.args[0] is image
    assert call.kwargs == {
        "hitThreshold": 1.0,
        "winStride": (4, 4),
        "padding": (8, 8),
        "scale": 1.05,
    }
