from typing import List

from pydantic import BaseModel


class BoundingBox(BaseModel):
    bbox: List[int]
    confidence: float


class DetectionResponse(BaseModel):
    filename: str
    raw_detections: int
    final_detections: int
    suppressed_detections: int
    latency_ms: float
    detections: List[BoundingBox]