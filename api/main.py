import logging
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from src.detector import HOGPedestrianDetector
from src.postprocessing import non_max_suppression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("pedestrian-api")

app = FastAPI(
    title="HOG Pedestrian Detection API",
    version="1.0.0",
)

detector = HOGPedestrianDetector()


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        response = await call_next(request)

        latency_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%.3f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )

        return response

    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000

        logger.exception(
            "request_id=%s method=%s path=%s status=500 latency_ms=%.3f",
            request_id,
            request.method,
            request.url.path,
            latency_ms,
        )

        raise


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "OpenCV HOG + SVM",
        "descriptor_size": detector.hog.getDescriptorSize(),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image",
        )

    result = detector.detect(image)

    final_detections = non_max_suppression(
        result["detections"],
        iou_threshold=0.3,
    )

    return {
        "filename": file.filename,
        "image": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        "raw_detections": result["count"],
        "final_detections": len(final_detections),
        "suppressed_detections": (
            result["count"] - len(final_detections)
        ),
        "latency_ms": result["latency_ms"],
        "detections": final_detections,
    }