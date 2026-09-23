import argparse
import json
import statistics
import time
from pathlib import Path

import cv2

from src.detector import HOGPedestrianDetector


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--runs", type=int, default=10)

    args = parser.parse_args()

    image = cv2.imread(args.input)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {args.input}"
        )

    detector = HOGPedestrianDetector()

    latencies = []
    detection_counts = []

    # Warm-up
    detector.detect(image)

    for _ in range(args.runs):
        start = time.perf_counter()

        result = detector.detect(image)

        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)
        detection_counts.append(result["count"])

    latencies_sorted = sorted(latencies)

    p50_index = int(0.50 * len(latencies_sorted))
    p95_index = min(
        int(0.95 * len(latencies_sorted)),
        len(latencies_sorted) - 1,
    )

    metrics = {
        "input": str(Path(args.input)),
        "runs": args.runs,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "mean_latency_ms": round(
            statistics.mean(latencies),
            3,
        ),
        "median_latency_ms": round(
            statistics.median(latencies),
            3,
        ),
        "p50_latency_ms": round(
            latencies_sorted[p50_index],
            3,
        ),
        "p95_latency_ms": round(
            latencies_sorted[p95_index],
            3,
        ),
        "min_latency_ms": round(
            min(latencies),
            3,
        ),
        "max_latency_ms": round(
            max(latencies),
            3,
        ),
        "mean_detections": round(
            statistics.mean(detection_counts),
            3,
        ),
        "detection_count_stable": (
            len(set(detection_counts)) == 1
        ),
    }

    print(json.dumps(metrics, indent=2))

    output_path = Path("metrics.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()