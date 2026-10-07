import argparse
import csv
import statistics

import cv2

from src.detector import HOGPedestrianDetector
from src.postprocessing import non_max_suppression


IOU_THRESHOLD = 0.5


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def normalize_name(name):
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(columns, candidates):
    normalized = {
        normalize_name(column): column
        for column in columns
    }

    for candidate in candidates:
        candidate = normalize_name(candidate)

        if candidate in normalized:
            return normalized[candidate]

    return None


def load_annotations(csv_path):
    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV has no header.")

        columns = reader.fieldnames

        print("CSV columns:")
        for column in columns:
            print(f"  {column}")

        x_col = find_column(
            columns,
            ["x", "x1", "xmin", "left", "bbox_x"],
        )

        y_col = find_column(
            columns,
            ["y", "y1", "ymin", "top", "bbox_y"],
        )

        width_col = find_column(
            columns,
            ["w", "width", "bbox_width"],
        )

        height_col = find_column(
            columns,
            ["h", "height", "bbox_height"],
        )

        if x_col is None or y_col is None:
            raise ValueError(
                "Could not identify x/y columns."
            )

        if width_col is None or height_col is None:
            raise ValueError(
                "Could not identify width/height columns."
            )

        annotations = {}

        # The CSV contains one bounding-box annotation
        # per video frame.
        #
        # Frame numbering is treated as 1-based:
        # first CSV row -> frame 1
        # second CSV row -> frame 2
        # etc.
        for frame_number, row in enumerate(
            reader,
            start=1,
        ):
            x = float(row[x_col])
            y = float(row[y_col])
            w = float(row[width_col])
            h = float(row[height_col])

            if w <= 0 or h <= 0:
                raise ValueError(
                    f"Invalid bounding box at annotation "
                    f"row {frame_number}: "
                    f"x={x}, y={y}, w={w}, h={h}"
                )

            box = [
                int(round(x)),
                int(round(y)),
                int(round(x + w)),
                int(round(y + h)),
            ]

            annotations[frame_number] = [box]

    return annotations


def match_predictions(predictions, ground_truth):
    predictions = sorted(
        predictions,
        key=lambda detection: detection["confidence"],
        reverse=True,
    )

    matched_gt = set()

    tp = 0
    fp = 0

    for prediction in predictions:
        x, y, width, height = prediction["bbox"]
        prediction_box = [x, y, x + width, y + height]

        best_iou = 0.0
        best_gt_index = None

        for index, gt_box in enumerate(ground_truth):
            if index in matched_gt:
                continue

            score = iou(
                prediction_box,
                gt_box,
            )

            if score > best_iou:
                best_iou = score
                best_gt_index = index

        if best_iou >= IOU_THRESHOLD:
            tp += 1
            matched_gt.add(best_gt_index)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video",
    )

    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to ground-truth CSV",
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Evaluate every Nth frame",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of evaluated frames",
    )

    args = parser.parse_args()

    if args.sample_every < 1:
        raise ValueError(
            "--sample-every must be >= 1"
        )

    if (
        args.max_frames is not None
        and args.max_frames < 1
    ):
        raise ValueError(
            "--max-frames must be >= 1"
        )

    ground_truth = load_annotations(
        args.annotations
    )

    detector = HOGPedestrianDetector()

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.video}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError(
            f"Invalid video FPS: {fps}"
        )

    print()
    print(f"Video: {args.video}")
    print(f"Reported frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"CSV annotation rows: {len(ground_truth)}")
    print(
        f"IoU threshold: {IOU_THRESHOLD}"
    )
    print()

    tp_total = 0
    fp_total = 0
    fn_total = 0

    evaluated_frames = 0
    inference_times = []
    annotated_frames = 0
    decoded_frames = 0

    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        current_frame_number = frame_index + 1
        decoded_frames += 1

        if current_frame_number in ground_truth:
            annotated_frames += 1

        if (
            current_frame_number
            % args.sample_every
            != 0
        ):
            frame_index += 1
            continue

        if current_frame_number not in ground_truth:
            frame_index += 1
            continue

        result = detector.detect(frame)

        detections = non_max_suppression(
            result["detections"],
            iou_threshold=(
                detector.config.nms_iou_threshold
            ),
        )

        inference_times.append(result["latency_ms"])

        gt_boxes = ground_truth[
            current_frame_number
        ]

        tp, fp, fn = match_predictions(
            detections,
            gt_boxes,
        )

        tp_total += tp
        fp_total += fp
        fn_total += fn

        evaluated_frames += 1

        if evaluated_frames % 25 == 0:
            print(
                f"Evaluated "
                f"{evaluated_frames} frames..."
            )

        if (
            args.max_frames is not None
            and evaluated_frames
            >= args.max_frames
        ):
            break

        frame_index += 1

    cap.release()

    if (
        decoded_frames != len(ground_truth)
        and (
            args.max_frames is None
            or evaluated_frames < args.max_frames
        )
    ):
        print(
            f"WARNING: decoded {decoded_frames} frames but "
            f"CSV contains {len(ground_truth)} annotations."
        )

    if evaluated_frames == 0:
        raise RuntimeError(
            "No annotated video frames were evaluated."
        )

    precision = (
        tp_total
        / (tp_total + fp_total)
        if (tp_total + fp_total)
        else 0.0
    )

    recall = (
        tp_total
        / (tp_total + fn_total)
        if (tp_total + fn_total)
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    mean_latency = (
        sum(inference_times)
        / len(inference_times)
        if inference_times
        else 0.0
    )
    ordered_latencies = sorted(inference_times)

    def percentile(percent):
        index = max(0, int((percent / 100) * len(ordered_latencies) + 0.999999) - 1)
        return ordered_latencies[index]

    print()
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(
        f"Total decoded frames: {decoded_frames}"
    )
    print(
        f"Annotated frames    : {annotated_frames}"
    )
    print(
        f"Frames evaluated : "
        f"{evaluated_frames}"
    )
    print(f"Sampling interval  : every {args.sample_every} frame(s)")

    print(
        f"TP                : "
        f"{tp_total}"
    )

    print(
        f"FP                : "
        f"{fp_total}"
    )

    print(
        f"FN                : "
        f"{fn_total}"
    )

    print(
        f"Precision         : "
        f"{precision:.4f}"
    )

    print(
        f"Recall            : "
        f"{recall:.4f}"
    )

    print(
        f"F1                : "
        f"{f1:.4f}"
    )

    print(
        f"Mean latency (ms) : "
        f"{mean_latency:.3f}"
    )
    print(f"Median latency (ms): {statistics.median(inference_times):.3f}")
    print(f"P95 latency (ms)   : {percentile(95):.3f}")
    print(f"Min latency (ms)   : {min(inference_times):.3f}")
    print(f"Max latency (ms)   : {max(inference_times):.3f}")

    print("=" * 50)


if __name__ == "__main__":
    main()
