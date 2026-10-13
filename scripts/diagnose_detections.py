"""Render a small, aligned visual sample of the pedestrian evaluation."""

import argparse
from pathlib import Path

import cv2

from scripts.evaluate_video import (
    IOU_THRESHOLD,
    load_annotations,
    match_predictions_with_details,
)
from src.detector import HOGPedestrianDetector
from src.postprocessing import non_max_suppression


ROOT = Path(__file__).resolve().parents[1]
CLIPS = ("crosswalk", "fourway", "night")


def select_frames(annotation_count):
    """Choose five unique 1-based frames spanning the annotated clip."""
    if annotation_count < 5:
        raise ValueError("At least five annotated frames are required")
    return [1 + round(index * (annotation_count - 1) / 4) for index in range(5)]


def _clip_box(box, width, height):
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    )


def draw_diagnostic(frame, clip, frame_number, ground_truth, detections, details):
    """Draw GT and post-NMS detections; clipping is display-only."""
    image = frame.copy()
    height, width = image.shape[:2]
    green = (0, 200, 0)
    red = (0, 0, 230)
    white = (255, 255, 255)

    matched_by_gt = {
        item["ground_truth_index"]: item
        for item in details["matches"]
    }
    for index, box in enumerate(ground_truth):
        x1, y1, x2, y2 = _clip_box(box, width, height)
        cv2.rectangle(image, (x1, y1), (x2, y2), green, 2)
        if index in matched_by_gt:
            label = f"GT {index + 1}: TP"
        else:
            label = f"GT {index + 1}: FN"
        cv2.putText(image, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, green, 2)

    matched_predictions = {
        id(item["prediction"]): item
        for item in details["matches"]
    }
    false_positive_ids = {id(item) for item in details["false_positives"]}
    for detection in detections:
        x, y, box_width, box_height = detection["bbox"]
        x1, y1, x2, y2 = _clip_box(
            [x, y, x + box_width, y + box_height], width, height
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), red, 2)
        if id(detection) in matched_predictions:
            item = matched_predictions[id(detection)]
            label = (
                f"TP conf={detection['confidence']:.2f} "
                f"IoU={item['iou']:.2f} GT={item['ground_truth_index'] + 1}"
            )
        elif id(detection) in false_positive_ids:
            label = f"FP conf={detection['confidence']:.2f}"
        else:
            raise AssertionError("Detection missing from matching result")
        cv2.putText(image, label, (x1, min(height - 5, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, red, 2)

    header = f"{clip} | frame {frame_number} | IoU threshold {IOU_THRESHOLD:.1f}"
    cv2.rectangle(image, (0, 0), (width, 32), (0, 0, 0), -1)
    cv2.putText(image, header, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
    return image


def generate_diagnostics(output_root=ROOT / "results" / "diagnostics"):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    detector = HOGPedestrianDetector()
    summary_lines = [
        "# Targeted detection diagnostics",
        "",
        "Five frames per clip were selected at evenly spaced 1-based annotation rows.",
        "Video frames were decoded sequentially; detector inference ran only on selected frames.",
        "Ground truth uses the evaluator CSV row mapping; predictions use configured detector and NMS.",
        f"Matching uses evaluator IoU threshold {IOU_THRESHOLD:.1f} and confidence-ordered greedy one-to-one matching.",
        "Ground-truth boxes are green; post-NMS predictions are red. Box clipping is for display only.",
        "",
    ]

    for clip in CLIPS:
        video_path = ROOT / "dataset" / f"{clip}.avi"
        csv_path = ROOT / "dataset" / f"{clip}.csv"
        annotations = load_annotations(csv_path)
        selected = set(select_frames(len(annotations)))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        clip_dir = output_root / clip
        clip_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        decoded = 0
        has_extra_frame = False
        try:
            while decoded < len(annotations):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"{clip}: video ended at decoded frame {decoded}; "
                        f"CSV has {len(annotations)} annotation rows"
                    )
                decoded += 1
                if decoded not in selected:
                    continue

                result = detector.detect(frame)
                detections = non_max_suppression(
                    result["detections"],
                    iou_threshold=detector.config.nms_iou_threshold,
                )
                truth = annotations[decoded]
                details = match_predictions_with_details(detections, truth)
                image = draw_diagnostic(frame, clip, decoded, truth, detections, details)
                image_path = clip_dir / f"frame_{decoded:06d}.jpg"
                if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                    raise RuntimeError(f"Could not write diagnostic image: {image_path}")
                tp = len(details["matches"])
                fp = len(details["false_positives"])
                fn = len(details["false_negatives"])
                rows.append((decoded, tp, fp, fn, image_path.relative_to(output_root).as_posix()))
            extra_ok, _ = cap.read()
            has_extra_frame = bool(extra_ok)
        finally:
            cap.release()

        if len(rows) != 5:
            raise RuntimeError(f"{clip}: expected 5 sampled frames, wrote {len(rows)}")
        alignment = "decoded frame count equals annotation rows"
        if has_extra_frame:
            alignment = "video has frames beyond annotation rows"

        summary_lines.extend([
            f"## {clip}",
            "",
            f"Annotation rows: {len(annotations)}; sequentially decoded through row {decoded}; {alignment}.",
            "",
            "| Frame | TP | FP | FN | Image |",
            "|---:|---:|---:|---:|---|",
        ])
        for frame_number, tp, fp, fn, image_rel in rows:
            summary_lines.append(f"| {frame_number} | {tp} | {fp} | {fn} | `{image_rel}` |")
        summary_lines.append("")

    summary_path = output_root / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "diagnostics",
        help="Output directory (default: results/diagnostics)",
    )
    args = parser.parse_args()
    print(generate_diagnostics(args.output_dir))


if __name__ == "__main__":
    main()
