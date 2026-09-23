import numpy as np

from src.detector import HOGPedestrianDetector


def test_detector_initializes():
    detector = HOGPedestrianDetector()

    assert detector.hog is not None
    assert detector.hog.getDescriptorSize() == 3780


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