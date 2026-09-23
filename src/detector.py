from time import perf_counter

import cv2

from src.config import DetectorConfig


class HOGPedestrianDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )

    def detect(self, image):
        if image is None:
            raise ValueError("Input image is None")

        if len(image.shape) != 3:
            raise ValueError("Input image must be a color image")

        start = perf_counter()

        boxes, weights = self.hog.detectMultiScale(
            image,
            hitThreshold=self.config.hit_threshold,
            winStride=self.config.win_stride,
            padding=self.config.padding,
            scale=self.config.scale,
        )

        latency_ms = (perf_counter() - start) * 1000

        detections = [
            {
                "bbox": [
                    int(x),
                    int(y),
                    int(w),
                    int(h),
                ],
                "confidence": float(weight),
            }
            for (x, y, w, h), weight in zip(boxes, weights)
        ]

        return {
            "detections": detections,
            "count": len(detections),
            "latency_ms": round(latency_ms, 3),
        }