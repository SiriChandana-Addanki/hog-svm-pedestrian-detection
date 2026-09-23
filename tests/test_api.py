from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


client = TestClient(app)


def create_test_image():
    image = Image.new("RGB", (320, 240), "white")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "OpenCV HOG + SVM"
    assert data["descriptor_size"] == 3780


def test_detect_image():
    image = create_test_image()

    response = client.post(
        "/detect",
        files={
            "file": (
                "test.png",
                image,
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "test.png"
    assert "raw_detections" in data
    assert "final_detections" in data
    assert "suppressed_detections" in data
    assert "latency_ms" in data
    assert "detections" in data


def test_reject_non_image():
    response = client.post(
        "/detect",
        files={
            "file": (
                "test.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400