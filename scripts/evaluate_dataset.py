import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd

from src.detector import HOGPedestrianDetector
from src.config import DetectorConfig


DATASETS = ["crosswalk", "fourway", "night"]

HIT_THRESHOLDS = [-0.5, 0.0, 0.5, 1.0, 1.5]

IOU_THRESHOLD = 0.5


def calculate_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def evaluate_detections(predictions, ground_truth, iou_threshold):
    matched_gt = set()

    tp = 0
    fp = 0
    fn = 0

    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(pred["bbox"], gt)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    return tp, fp, fn


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0

    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def create_detector(hit_threshold):
    config = DetectorConfig(
        hit_threshold=hit_threshold,
    )

    return HOGPedestrianDetector(config)


def evaluate_dataset(name):
    video_path = f"dataset/{name}.avi"
    csv_path = f"dataset/{name}.csv"

    annotations = pd.read_csv(csv_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    detectors = {
        hit_threshold: create_detector(hit_threshold)
        for hit_threshold in HIT_THRESHOLDS
    }

    results = {
        hit_threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for hit_threshold in HIT_THRESHOLDS
    }

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        print(
            f"\r{name}: frame {frame_count + 1}/{len(annotations)}",
            end="",
            flush=True,
        )

        if frame_count >= len(annotations):
            raise RuntimeError(
                f"{name}: more video frames than CSV annotations"
            )

        row = annotations.iloc[frame_count]

        ground_truth = [
            [
                int(row["x"]),
                int(row["y"]),
                int(row["w"]),
                int(row["h"]),
            ]
        ]

        for hit_threshold in HIT_THRESHOLDS:
            detector = detectors[hit_threshold]

            result = detector.detect(frame)

            predictions = result["detections"]

            frame_tp, frame_fp, frame_fn = evaluate_detections(
                predictions,
                ground_truth,
                IOU_THRESHOLD,
            )

            results[hit_threshold]["tp"] += frame_tp
            results[hit_threshold]["fp"] += frame_fp
            results[hit_threshold]["fn"] += frame_fn

        frame_count += 1

    cap.release()

    if frame_count != len(annotations):
        raise RuntimeError(
            f"{name}: decoded {frame_count} frames but "
            f"found {len(annotations)} CSV rows"
        )

    metrics = {}

    for hit_threshold in HIT_THRESHOLDS:
        tp = results[hit_threshold]["tp"]
        fp = results[hit_threshold]["fp"]
        fn = results[hit_threshold]["fn"]

        precision, recall, f1 = calculate_metrics(
            tp,
            fp,
            fn,
        )

        metrics[hit_threshold] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "dataset": name,
        "frames": frame_count,
        "metrics": metrics,
    }


def main():
    overall = {
        hit_threshold: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
        }
        for hit_threshold in HIT_THRESHOLDS
    }

    for name in DATASETS:
        print(f"\nEvaluating {name}...")

        result = evaluate_dataset(name)

        print(f"Frames: {result['frames']}")

        for hit_threshold in HIT_THRESHOLDS:
            metrics = result["metrics"][hit_threshold]

            print(
                f"hitThreshold {hit_threshold:.1f} | "
                f"TP={metrics['TP']} "
                f"FP={metrics['FP']} "
                f"FN={metrics['FN']} | "
                f"Precision={metrics['precision']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f}"
            )

            overall[hit_threshold]["TP"] += metrics["TP"]
            overall[hit_threshold]["FP"] += metrics["FP"]
            overall[hit_threshold]["FN"] += metrics["FN"]

    print("\n========== OVERALL ==========")

    for hit_threshold in HIT_THRESHOLDS:
        tp = overall[hit_threshold]["TP"]
        fp = overall[hit_threshold]["FP"]
        fn = overall[hit_threshold]["FN"]

        precision, recall, f1 = calculate_metrics(
            tp,
            fp,
            fn,
        )

        print(
            f"hitThreshold {hit_threshold:.1f} | "
            f"TP={tp} "
            f"FP={fp} "
            f"FN={fn} | "
            f"Precision={precision:.4f} "
            f"Recall={recall:.4f} "
            f"F1={f1:.4f}"
        )


if __name__ == "__main__":
    main()