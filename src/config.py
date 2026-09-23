from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    win_stride: tuple[int, int] = (4, 4)
    padding: tuple[int, int] = (8, 8)
    scale: float = 1.05
    hit_threshold: float = 0.0
    nms_iou_threshold: float = 0.3