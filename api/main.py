from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.detector import HOGPedestrianDetector
from src.postprocessing import non_max_suppression

app = FastAPI(
    title="HOG Pedestrian Detection API",
    version="1.0.0",
)

detector = HOGPedestrianDetector()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "OpenCV HOG + SVM",
        "descriptor_size": detector.hog.getDescriptorSize(),
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):
    if not file.content_type or not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image",
        )

    contents = await file.read()

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