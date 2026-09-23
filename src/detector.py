from pathlib import Path
from time import perf_counter

import cv2


class HOGPedestrianDetector:
    def __init__(
        self,
        win_stride=(4, 4),
        padding=(8, 8),
        scale=1.05,
        hit_threshold=0.0,
    ):
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.hit_threshold = hit_threshold

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )

    def detect(self, image):
        if image is None:
            raise ValueError("Input image is None")

        if not isinstance(image, (cv2.UMat,)) and len(image.shape) != 3:
            raise ValueError("Input image must be a color image")

        start = perf_counter()

        boxes, weights = self.hog.detectMultiScale(
            image,
            hitThreshold=self.hit_threshold,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
        )

        latency_ms = (perf_counter() - start) * 1000

        detections = []

        for (x, y, w, h), weight in zip(boxes, weights):
            detections.append(
                {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "confidence": float(weight),
                }
            )

        return {
            "detections": detections,
            "count": len(detections),
            "latency_ms": round(latency_ms, 3),
        }