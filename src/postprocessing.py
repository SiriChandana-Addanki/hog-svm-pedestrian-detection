import numpy as np


def non_max_suppression(detections, iou_threshold=0.3):
    if not detections:
        return []

    boxes = np.array(
        [d["bbox"] for d in detections],
        dtype=np.float32,
    )

    scores = np.array(
        [d["confidence"] for d in detections],
        dtype=np.float32,
    )

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        width = np.maximum(0.0, xx2 - xx1)
        height = np.maximum(0.0, yy2 - yy1)

        intersection = width * height

        union = (
            areas[i]
            + areas[order[1:]]
            - intersection
        )

        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )

        remaining = np.where(iou <= iou_threshold)[0]

        order = order[remaining + 1]

    return [detections[i] for i in keep]