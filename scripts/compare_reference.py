"""Compare the current HOG setup with the supplied resize/padding reference."""

import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

import cv2

from src.postprocessing import non_max_suppression
from scripts.evaluate_video import load_annotations, match_predictions


VARIANTS = (
    ("current_original_pad8", None, (8, 8), 0.0),
    ("original_pad4", None, (4, 4), 0.0),
    ("resize400_pad8", 400, (8, 8), 0.0),
    ("reference_resize400_pad4", 400, (4, 4), 0.0),
    ("current_hit0_5", None, (8, 8), 0.5),
    ("current_hit1_0", None, (8, 8), 1.0),
)


def run_variant(frame, resize_width, padding, hit_threshold):
    original_height, original_width = frame.shape[:2]
    target_width = min(resize_width, original_width) if resize_width else original_width
    scale = target_width / original_width
    target_height = int(original_height * scale)

    preprocess_start = perf_counter()
    if scale < 1:
        image = cv2.resize(frame, (target_width, target_height))
    else:
        image = frame
    preprocess_ms = (perf_counter() - preprocess_start) * 1000

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cpu_start = process_time()
    hog_start = perf_counter()
    boxes, weights = hog.detectMultiScale(
        image,
        hitThreshold=hit_threshold,
        winStride=(4, 4),
        padding=padding,
        scale=1.05,
    )
    hog_ms = (perf_counter() - hog_start) * 1000
    cpu_ms = (process_time() - cpu_start) * 1000
    input_boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]

    detections = []
    for (x, y, width, height), weight in zip(boxes, weights):
        x1 = int(round(int(x) / scale))
        y1 = int(round(int(y) / scale))
        x2 = int(round((int(x) + int(width)) / scale))
        y2 = int(round((int(y) + int(height)) / scale))
        detections.append(
            {
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "confidence": float(weight),
            }
        )

    final = non_max_suppression(detections, iou_threshold=0.3)
    return {
        "detections": detections,
        "final": final,
        "preprocess_ms": preprocess_ms,
        "hog_ms": hog_ms,
        "wall_ms": preprocess_ms + hog_ms,
        "cpu_ms": cpu_ms,
        "target_size": [target_width, target_height],
        "scale": scale,
        "input_boxes": input_boxes,
    }


def annotate(frame, predictions, ground_truth=None):
    output = frame.copy()
    if ground_truth:
        for x1, y1, x2, y2 in ground_truth:
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for detection in predictions:
        x, y, width, height = detection["bbox"]
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 2)
    return output


def compare_frame(frame, name, output_dir, frame_number=None, ground_truth=None):
    reports = []
    prefix = f"{name}_frame_{frame_number}" if frame_number is not None else name
    for variant, resize_width, padding, hit_threshold in VARIANTS:
        result = run_variant(frame, resize_width, padding, hit_threshold)
        raw = result["detections"]
        final = result["final"]
        report = {
            "variant": variant,
            "input_size": [int(frame.shape[1]), int(frame.shape[0])],
            "detector_input_size": result["target_size"],
            "win_stride": [4, 4],
            "padding": list(padding),
            "scale": 1.05,
            "hit_threshold": hit_threshold,
            "resize_scale": result["scale"],
            "preprocess_ms": round(result["preprocess_ms"], 3),
            "hog_ms": round(result["hog_ms"], 3),
            "detector_wall_ms": round(result["wall_ms"], 3),
            "process_cpu_ms": round(result["cpu_ms"], 3),
            "raw_count": len(raw),
            "nms_count": len(final),
            "confidence_min": min((d["confidence"] for d in raw), default=None),
            "confidence_median": (
                sorted(d["confidence"] for d in raw)[len(raw) // 2]
                if raw
                else None
            ),
            "confidence_max": max((d["confidence"] for d in raw), default=None),
        }
        if ground_truth is not None:
            tp, fp, fn = match_predictions(final, ground_truth)
            report.update(tp=tp, fp=fp, fn=fn)

        image_path = output_dir / f"{prefix}_{variant}.png"
        if not cv2.imwrite(str(image_path), annotate(frame, final, ground_truth)):
            raise RuntimeError(f"Could not write {image_path}")
        report["visual"] = str(image_path)
        reports.append(report)

        if variant == "reference_resize400_pad4" and frame_number is None:
            reference_image = cv2.resize(frame, tuple(result["target_size"]))
            for x, y, width, height in result["input_boxes"]:
                cv2.rectangle(
                    reference_image,
                    (x, y),
                    (x + width, y + height),
                    (0, 0, 255),
                    2,
                )
            cv2.imwrite(str(output_dir / "reference_exact_raw_resized.png"), reference_image)

    return reports


def extract_frames(video_path, frame_numbers):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    wanted = set(frame_numbers)
    frames = {}
    frame_number = 0
    while wanted - frames.keys():
        ok, frame = capture.read()
        if not ok:
            break
        frame_number += 1
        if frame_number in wanted:
            frames[frame_number] = frame
    capture.release()
    missing = sorted(wanted - frames.keys())
    if missing:
        raise RuntimeError(f"Video ended before requested frames: {missing}")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Image to compare")
    source.add_argument("--video", help="Video to sample")
    parser.add_argument("--annotations", help="CSV annotations for --video")
    parser.add_argument("--frames", help="Comma-separated 1-based frames (default: first, middle, last)")
    parser.add_argument("--output-dir", default="reference_comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        frame = cv2.imread(args.input)
        if frame is None:
            raise FileNotFoundError(args.input)
        reports = compare_frame(frame, Path(args.input).stem, output_dir)
    else:
        if not args.annotations:
            parser.error("--annotations is required with --video")
        annotations = load_annotations(args.annotations)
        if args.frames:
            frame_numbers = sorted({int(item) for item in args.frames.split(",")})
        else:
            frame_numbers = sorted({1, (len(annotations) + 1) // 2, len(annotations)})
        for number in frame_numbers:
            if number not in annotations:
                raise ValueError(f"No annotation for frame {number}")
        frames = extract_frames(args.video, frame_numbers)
        reports = []
        for number in frame_numbers:
            gt = annotations[number]
            reports.extend(
                compare_frame(frames[number], Path(args.video).stem, output_dir, number, gt)
            )

    output_path = output_dir / "comparison.json"
    output_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
