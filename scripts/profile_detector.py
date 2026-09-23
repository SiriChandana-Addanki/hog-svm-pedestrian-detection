import argparse
import json
import statistics
import time
from pathlib import Path

import cv2

from src.detector import HOGPedestrianDetector
from src.config import DetectorConfig


CONFIGS = [
    {
        "name": "baseline",
        "win_stride": (4, 4),
        "padding": (8, 8),
        "scale": 1.05,
    },
    {
        "name": "stride_8",
        "win_stride": (8, 8),
        "padding": (8, 8),
        "scale": 1.05,
    },
    {
        "name": "scale_1_10",
        "win_stride": (4, 4),
        "padding": (8, 8),
        "scale": 1.10,
    },
    {
        "name": "stride_8_scale_1_10",
        "win_stride": (8, 8),
        "padding": (8, 8),
        "scale": 1.10,
    },
]


def benchmark(detector, image, runs):
    detector.detect(image)

    latencies = []
    detection_counts = []

    for _ in range(runs):
        start = time.perf_counter()

        result = detector.detect(image)

        latency_ms = (time.perf_counter() - start) * 1000

        latencies.append(latency_ms)
        detection_counts.append(result["count"])

    return {
        "mean_latency_ms": round(
            statistics.mean(latencies), 3
        ),
        "p95_latency_ms": round(
            sorted(latencies)[
                min(int(runs * 0.95), runs - 1)
            ],
            3,
        ),
        "min_latency_ms": round(min(latencies), 3),
        "max_latency_ms": round(max(latencies), 3),
        "mean_detections": round(
            statistics.mean(detection_counts), 3
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", default="profile_results.json")
    args = parser.parse_args()

    image = cv2.imread(args.input)

    if image is None:
        raise FileNotFoundError(args.input)

    results = []

    for config in CONFIGS:
        detector = HOGPedestrianDetector(
    DetectorConfig(
        win_stride=config["win_stride"],
        padding=config["padding"],
        scale=config["scale"],
    )
)

        metrics = benchmark(
            detector,
            image,
            args.runs,
        )

        results.append(
            {
                **config,
                **metrics,
            }
        )

    output = {
        "input": args.input,
        "runs": args.runs,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "configs": results,
    }

    Path(args.output).write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()