import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd

from src.detector import HOGPedestrianDetector


DATASET = "crosswalk"
FRAME_NUMBERS = [1, 100, 200, 300, 378]

OUTPUT_DIR = Path("evaluation_visuals")
OUTPUT_DIR.mkdir(exist_ok=True)

detector = HOGPedestrianDetector()

video_path = f"dataset/{DATASET}.avi"
csv_path = f"dataset/{DATASET}.csv"

annotations = pd.read_csv(csv_path)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {video_path}")

frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    if frame_number not in FRAME_NUMBERS:
        continue

    row = annotations.iloc[frame_number - 1]

    gt_x = int(row["x"])
    gt_y = int(row["y"])
    gt_w = int(row["w"])
    gt_h = int(row["h"])

    # Ground truth = GREEN
    cv2.rectangle(
        frame,
        (gt_x, gt_y),
        (gt_x + gt_w, gt_y + gt_h),
        (0, 255, 0),
        3,
    )

    result = detector.detect(frame)

    predictions = result["detections"]

    # Ground truth = GREEN
    cv2.rectangle(
    frame,
    (gt_x, gt_y),
    (gt_x + gt_w, gt_y + gt_h),
    (0, 255, 0),
    3,
)

    # Predictions = RED
    for detection in predictions:
        x, y, w, h = detection["bbox"]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2,
        )

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    output_path = OUTPUT_DIR / f"{DATASET}_frame_{frame_number}.jpg"

    cv2.imwrite(str(output_path), frame)

    print(f"Saved: {output_path}")

cap.release()

print("\nDone.")