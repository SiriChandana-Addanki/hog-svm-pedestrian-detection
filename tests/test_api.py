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
    assert response.headers.get("X-Request-ID")

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
    assert "raw_detection_results" in data
    assert data["image"] == {"width": 320, "height": 240}
    assert data["request_id"] == response.headers["X-Request-ID"]


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


def test_reject_empty_image():
    response = client.post(
        "/detect",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"


def test_reject_undecodable_image():
    response = client.post(
        "/detect",
        files={"file": ("broken.png", b"not really png", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Could not decode image"
