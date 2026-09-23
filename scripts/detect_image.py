import argparse
import json
from pathlib import Path

import cv2

from src.detector import HOGPedestrianDetector
from src.postprocessing import non_max_suppression


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.3)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    detector = HOGPedestrianDetector()

    baseline = detector.detect(image)

    final_detections = non_max_suppression(
        baseline["detections"],
        iou_threshold=args.iou_threshold,
    )

    output_image = image.copy()

    for detection in final_detections:
        x, y, w, h = detection["bbox"]

        cv2.rectangle(
            output_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output_image,
            f'{detection["confidence"]:.2f}',
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_path),
        output_image,
    ):
        raise RuntimeError(
            f"Could not write output: {output_path}"
        )

    result = {
        "input": str(input_path),
        "iou_threshold": args.iou_threshold,
        "raw_detections": baseline["count"],
        "final_detections": len(final_detections),
        "suppressed_detections": (
            baseline["count"] - len(final_detections)
        ),
        "suppression_ratio": round(
            (
                baseline["count"] - len(final_detections)
            )
            / baseline["count"],
            4,
        )
        if baseline["count"] > 0
        else 0.0,
        "latency_ms": baseline["latency_ms"],
    }

    print(json.dumps(result, indent=2))
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()