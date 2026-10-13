"""Compare two HOG scale values on a stratified 60-frame diagnostic sample."""

import argparse
import time
import warnings
from dataclasses import replace
from pathlib import Path

import cv2

from scripts.diagnose_detections import ROOT, draw_diagnostic
from scripts.evaluate_video import (
    IOU_THRESHOLD,
    iou,
    load_annotations,
    match_predictions_with_details,
)
from src.annotation_validation import warn_if_extreme_annotation
from src.config import DetectorConfig
from src.detector import HOGPedestrianDetector
from src.postprocessing import non_max_suppression


CLIPS = ("crosswalk", "fourway", "night")
BASELINE = 1.05
CANDIDATE = 1.08
ANOMALOUS_FOURWAY_ROW = 1

# The original 5-frame visual sample and the first 20-frame-per-clip sample.
PRIOR_SAMPLES = {
    "crosswalk": {1, 21, 41, 61, 80, 95, 100, 120, 140, 160, 180, 189, 199, 219, 239, 259, 279, 284, 299, 318, 338, 358, 378},
    "fourway": {1, 68, 136, 203, 270, 321, 338, 405, 473, 540, 607, 641, 675, 742, 809, 877, 944, 961, 1012, 1079, 1146, 1214, 1281},
    "night": {1, 31, 60, 90, 120, 142, 149, 179, 209, 238, 268, 283, 298, 328, 357, 387, 417, 424, 446, 476, 506, 535, 565},
}


def select_frames(annotation_count, count=20):
    if annotation_count < count:
        raise ValueError(f"Need at least {count} annotation rows, got {annotation_count}")
    selected = [1 + round(i * (annotation_count - 1) / (count - 1)) for i in range(count)]
    if len(set(selected)) != count:
        raise ValueError("Frame selection did not produce unique frames")
    return selected


def select_disjoint_frames(annotation_count, forbidden_frames, count=20):
    """Select evenly spread unique 1-based rows outside prior samples."""
    eligible = [frame for frame in range(1, annotation_count + 1) if frame not in forbidden_frames]
    if len(eligible) < count:
        raise ValueError(f"Need {count} eligible rows, found {len(eligible)}")
    selected = [eligible[round(i * (len(eligible) - 1) / (count - 1))] for i in range(count)]
    if len(set(selected)) != count or set(selected) & set(forbidden_frames):
        raise ValueError("Disjoint frame selection is not unique or overlaps exclusions")
    return selected


def metrics(details):
    tp = len(details["matches"])
    fp = len(details["false_positives"])
    fn = len(details["false_negatives"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def fmt_metrics(m):
    return f"{m['tp']}/{m['fp']}/{m['fn']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f}"


def run(output_dir, *, disjoint=False, preflight_only=False):
    started_all = time.perf_counter()
    selection = {}
    warnings_seen = []

    # Preflight annotation geometry against actual decoded frames before any scoring.
    for clip in CLIPS:
        annotations = load_annotations(ROOT / "dataset" / f"{clip}.csv")
        selected = (
            select_disjoint_frames(len(annotations), PRIOR_SAMPLES[clip])
            if disjoint else select_frames(len(annotations))
        )
        selection[clip] = selected
        cap = cv2.VideoCapture(str(ROOT / "dataset" / f"{clip}.avi"))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for {clip}")
        print(f"PREFLIGHT {clip}: selected frames {selected}")
        decoded = 0
        try:
            while decoded < max(selected):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"{clip}: decode ended at frame {decoded + 1}")
                decoded += 1
                if decoded not in selected:
                    continue
                for box in annotations[decoded]:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        warn_if_extreme_annotation(decoded, box, frame.shape, source=f"dataset/{clip}.csv")
                    for warning in caught:
                        message = str(warning.message)
                        warnings_seen.append(message)
                        print(f"ANNOTATION WARNING: {message}")
        finally:
            cap.release()

    if preflight_only:
        return selection, warnings_seen

    if disjoint:
        for clip in CLIPS:
            overlap = set(selection[clip]) & PRIOR_SAMPLES[clip]
            if overlap:
                raise AssertionError(f"{clip}: selected rows overlap prior samples: {sorted(overlap)}")

    output_dir.mkdir(parents=True, exist_ok=False)
    configs = {
        "1.05": HOGPedestrianDetector(replace(DetectorConfig(), scale=BASELINE)),
        "1.08": HOGPedestrianDetector(replace(DetectorConfig(), scale=CANDIDATE)),
    }
    records = {key: {} for key in configs}
    clip_elapsed = {clip: {key: 0.0 for key in configs} for clip in CLIPS}
    clip_decode_elapsed = {}

    # Re-decode selected frames once, then score the same decoded frame under both scales.
    for clip in CLIPS:
        annotations = load_annotations(ROOT / "dataset" / f"{clip}.csv")
        selected = set(selection[clip])
        cap = cv2.VideoCapture(str(ROOT / "dataset" / f"{clip}.avi"))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for {clip}")
        decode_elapsed = 0.0
        decoded = 0
        retained_frames = {}
        try:
            while decoded < max(selected):
                decode_start = time.perf_counter()
                ok, frame = cap.read()
                decode_elapsed += time.perf_counter() - decode_start
                if not ok:
                    raise RuntimeError(f"{clip}: decode ended at frame {decoded + 1}")
                decoded += 1
                if decoded not in selected:
                    continue
                retained_frames[decoded] = frame
                truth = annotations[decoded]
                for label, detector in configs.items():
                    call_start = time.perf_counter()
                    result = detector.detect(frame)
                    detections = non_max_suppression(
                        result["detections"],
                        iou_threshold=detector.config.nms_iou_threshold,
                    )
                    details = match_predictions_with_details(detections, truth)
                    clip_elapsed[clip][label] += time.perf_counter() - call_start
                    records[label][(clip, decoded)] = {
                        "raw": len(result["detections"]),
                        "post": len(detections),
                        "wall_ms": result["latency_ms"],
                        "cpu_ms": result["cpu_time_ms"],
                        "detections": detections,
                        "details": details,
                        "all_details": details,
                        "ground_truth": truth,
                    }
        finally:
            cap.release()
        clip_decode_elapsed[clip] = decode_elapsed

        # Drop decoded image memory after writing the representative overlays below.
        _write_clip_overlays(
            clip, retained_frames, records, annotations, output_dir,
            changed_only=disjoint,
        )
        del retained_frames

    report = _make_report(selection, records, clip_elapsed, clip_decode_elapsed, warnings_seen, disjoint=disjoint)
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Total elapsed including shared decode, inference, scoring, and output: {time.perf_counter() - started_all:.2f} s")


def _write_clip_overlays(clip, frames, records, annotations, output_dir, *, changed_only=False):
    # Prefer one example for each observed category; same frame rendered at both scales.
    keys = [(clip, frame) for frame in frames]
    recovered = lost = fp_example = None
    changed = []
    for key in keys:
        if clip == "fourway" and key[1] == ANOMALOUS_FOURWAY_ROW:
            continue
        b = records["1.05"][key]["details"]
        c = records["1.08"][key]["details"]
        if _match_signature(b) != _match_signature(c):
            changed.append(key)
        if recovered is None and len(c["matches"]) > len(b["matches"]):
            recovered = key
        if lost is None and len(b["matches"]) > len(c["matches"]):
            lost = key
        if fp_example is None and (b["false_positives"] or c["false_positives"]):
            fp_example = key
    chosen = []
    for kind, key in (("recovered", recovered), ("lost", lost), ("false_positive", fp_example)):
        if key and key not in [pair[1] for pair in chosen]:
            chosen.append((kind, key[1]))
    clip_dir = output_dir / clip
    clip_dir.mkdir(parents=True, exist_ok=True)
    if changed_only:
        for _, frame_number in changed:
            for label in ("1.05", "1.08"):
                entry = records[label][(clip, frame_number)]
                image = draw_diagnostic(
                    frames[frame_number], clip, frame_number, annotations[frame_number],
                    entry["detections"], entry["details"],
                )
                height, width = image.shape[:2]
                cv2.rectangle(image, (0, 32), (width, 62), (0, 0, 0), -1)
                cv2.putText(image, f"outcome differs | scale={label}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                path = clip_dir / f"frame_{frame_number:06d}_scale_{label}.jpg"
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                    raise RuntimeError(f"Could not write {path}")
        return
    for kind, frame_number in chosen:
        for label in ("1.05", "1.08"):
            entry = records[label][(clip, frame_number)]
            image = draw_diagnostic(
                frames[frame_number], clip, frame_number, annotations[frame_number],
                entry["detections"], entry["details"],
            )
            height, width = image.shape[:2]
            cv2.rectangle(image, (0, 32), (width, 62), (0, 0, 0), -1)
            cv2.putText(image, f"{kind} comparison | scale={label}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            path = clip_dir / f"frame_{frame_number:06d}_{kind}_scale_{label}.jpg"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError(f"Could not write {path}")


def _match_signature(details):
    """Compare matched GT assignments and unmatched GT/prediction sets."""
    matches = tuple(sorted(item["ground_truth_index"] for item in details["matches"]))
    false_positives = len(details["false_positives"])
    false_negatives = tuple(sorted(item["ground_truth_index"] for item in details["false_negatives"]))
    return matches, false_positives, false_negatives


def _make_report(selection, records, clip_elapsed, clip_decode_elapsed, warnings_seen, *, disjoint=False):
    lines = [
        "# Scale 1.08 validation: disjoint diagnostic sample" if disjoint else "# Scale 1.08 validation: stratified diagnostic sample", "",
        "Diagnostic subset only; not full-dataset performance. Each clip contributes 20 evenly distributed 1-based annotation rows (60 frames inferred under each setting). The same decoded frame is passed to both configurations.",
        "", "Only `scale` changes: 1.05 vs 1.08. `hit_threshold=1.0`, stride (4,4), padding (8,8), NMS IoU 0.3, and evaluator one-to-one confidence-ordered IoU >= 0.5 matching remain fixed.",
        "", ("The frame selector excludes every frame from the original 15-frame sample and the preceding 60-frame sample. Known anomalous Fourway row 1 is also outside the selected set, so no annotation was excluded from this primary comparison. All selected warning-flagged annotations remain included." if disjoint else "Metrics exclude only the known anomalous Fourway annotation row/frame 1. That frame was still inferred under both settings and its unchanged annotation comparison is reported separately. No other selected annotations are omitted. Raw/post-NMS counts in primary totals exclude that frame to keep denominators aligned."),
        "", ("Detector-call wall totals cover all 20 selected frames per clip (60 calls per setting). Per-clip experiment elapsed also includes NMS and matching; shared decode is reported separately." if disjoint else "Detector-call wall totals align with primary metrics: 20 Crosswalk + 19 Fourway + 20 Night calls; Fourway row 1 was inferred but its detector-call timing is omitted from primary totals with the known annotation anomaly. The per-clip experiment elapsed is a broader stopwatch that includes all 20 frames, including that row, plus NMS and matching; shared decode is reported separately. Combined detector-call wall time therefore covers 59 scored frames, while combined experiment elapsed covers all 60 inference frames."),
        "", "## Aggregate metrics", "", "TP/FP/FN | Precision | Recall | F1", "", "| Clip | Scale | TP/FP/FN | Precision | Recall | F1 | Raw detections | Post-NMS | Detector wall ms | Experiment s | Shared decode s |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for clip in (*CLIPS, "COMBINED"):
        use_clips = CLIPS if clip == "COMBINED" else (clip,)
        clip_keys = [(c, f) for c in use_clips for f in selection[c] if not (c == "fourway" and f == ANOMALOUS_FOURWAY_ROW)]
        for label in ("1.05", "1.08"):
            details = [records[label][key]["details"] for key in clip_keys]
            agg = {k: sum(metrics(d)[k] for d in details) for k in ("tp", "fp", "fn")}
            agg["precision"] = agg["tp"] / (agg["tp"] + agg["fp"]) if agg["tp"] + agg["fp"] else 0.0
            agg["recall"] = agg["tp"] / (agg["tp"] + agg["fn"]) if agg["tp"] + agg["fn"] else 0.0
            agg["f1"] = 2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"]) if agg["precision"] + agg["recall"] else 0.0
            raw = sum(records[label][k]["raw"] for k in clip_keys)
            post = sum(records[label][k]["post"] for k in clip_keys)
            detector_ms = sum(records[label][k]["wall_ms"] for k in clip_keys)
            exp_s = sum(clip_elapsed[c][label] for c in CLIPS) if clip == "COMBINED" else clip_elapsed[clip][label]
            decode_s = sum(clip_decode_elapsed.values()) if clip == "COMBINED" else clip_decode_elapsed[clip]
            lines.append(f"| {clip} | {label} | {agg['tp']}/{agg['fp']}/{agg['fn']} | {agg['precision']:.4f} | {agg['recall']:.4f} | {agg['f1']:.4f} | {raw} | {post} | {detector_ms:.1f} | {exp_s:.2f} | {decode_s:.3f} |")

    fourway_key = ("fourway", ANOMALOUS_FOURWAY_ROW)
    if ANOMALOUS_FOURWAY_ROW in selection["fourway"]:
        lines.extend(["", "## Known anomalous Fourway row 1 (separate, not in primary totals)", "", "| Scale | TP | FP | FN | Raw | Post-NMS |", "|---:|---:|---:|---:|---:|---:|"])
        for label in ("1.05", "1.08"):
            entry = records[label][fourway_key]
            m = metrics(entry["details"])
            lines.append(f"| {label} | {m['tp']} | {m['fp']} | {m['fn']} | {entry['raw']} | {entry['post']} |")

    lines.extend(["", "## Annotation preflight", ""])
    for clip in CLIPS:
        lines.append(f"- **{clip}:** frames `{', '.join(map(str, selection[clip]))}`.")
    lines.append(f"- Warning-flagged selected annotations: {len(warnings_seen)}. All selected flagged rows remain included; any explicit exclusion is listed above.")
    for warning in warnings_seen:
        lines.append(f"  - `{warning}`")

    frame_intro = "Rows list both scales as `TP/FP/FN`, raw->post-NMS, and matched IoUs. FP means unmatched by the evaluator at IoU >= 0.5; visual FP examples can still depict a pedestrian whose predicted box is poorly localized. No selected annotation rows were excluded." if disjoint else "Rows list both scales as `TP/FP/FN`, raw->post-NMS, and matched IoUs. FP means unmatched by the evaluator at IoU >= 0.5; visual FP examples can still depict a pedestrian whose predicted box is poorly localized. Anomaly row 1 remains shown separately and excluded from primary analysis."
    lines.extend(["", "## Per-frame comparison", "", frame_intro, "", "| Clip | Frame | GT | Scale 1.05 TP/FP/FN | raw->post | matched IoU | Scale 1.08 TP/FP/FN | raw->post | matched IoU | Change |", "|---|---:|---:|---|---:|---|---|---:|---|---|"])
    for clip in CLIPS:
        for frame in selection[clip]:
            key = (clip, frame)
            b, c = records["1.05"][key], records["1.08"][key]
            mb, mc = metrics(b["details"]), metrics(c["details"])
            iob = ",".join(f"{m['iou']:.3f}" for m in b["details"]["matches"]) or "-"
            ioc = ",".join(f"{m['iou']:.3f}" for m in c["details"]["matches"]) or "-"
            anomaly = clip == "fourway" and frame == ANOMALOUS_FOURWAY_ROW
            change = "excluded anomaly" if anomaly else ("changed" if (mb["tp"], mb["fp"], mb["fn"]) != (mc["tp"], mc["fp"], mc["fn"]) else "same counts")
            lines.append(f"| {clip} | {frame} | {len(b['details']['matches']) + len(b['details']['false_negatives'])} | {mb['tp']}/{mb['fp']}/{mb['fn']} | {b['raw']}->{b['post']} | {iob} | {mc['tp']}/{mc['fp']}/{mc['fn']} | {c['raw']}->{c['post']} | {ioc} | {change} |")

    lines.extend(["", "## Changed-frame best-overlap details", "", "For each frame where TP/FP/FN counts changed, this is the maximum IoU of any post-NMS prediction with the frame's ground truth, including predictions below the 0.5 match cutoff.", "", "| Clip | Frame | Baseline best prediction IoU | Candidate best prediction IoU | Outcome |", "|---|---:|---:|---:|---|"])
    for clip in CLIPS:
        for frame in selection[clip]:
            if clip == "fourway" and frame == ANOMALOUS_FOURWAY_ROW:
                continue
            key = (clip, frame)
            b, c = records["1.05"][key], records["1.08"][key]
            mb, mc = metrics(b["details"]), metrics(c["details"])
            if (mb["tp"], mb["fp"], mb["fn"]) == (mc["tp"], mc["fp"], mc["fn"]):
                continue
            def best_iou(entry):
                values = []
                for prediction in entry["detections"]:
                    x, y, width, height = prediction["bbox"]
                    pred_box = [x, y, x + width, y + height]
                    values.extend(iou(pred_box, gt) for gt in entry["ground_truth"])
                return max(values) if values else 0.0
            outcome = "recovered TP" if mc["tp"] > mb["tp"] else "lost TP" if mc["tp"] < mb["tp"] else "FP count changed"
            lines.append(f"| {clip} | {frame} | {best_iou(b):.3f} | {best_iou(c):.3f} | {outcome} |")
    lines.extend(["", "A recovered match means a candidate-scale match on ground truth that is unmatched at baseline; a lost match is the reverse. Review the paired overlays in this directory before drawing localization conclusions.", "", "## Visual overlays", ""])
    overlay_text = "Paired overlays are saved only for frames where the matched-GT/FP/FN outcome signature differs; frame numbers are in the file names." if disjoint else "Paired baseline/candidate overlays are saved under the clip subdirectories; filenames identify recovered, lost, or false-positive examples when such cases exist in that clip."
    lines.append(overlay_text)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "diagnostics" / "scale_1_08_validation")
    parser.add_argument("--disjoint", action="store_true", help="Use a new sample disjoint from the original 15 and prior 60 frames; save overlays only where match outcomes differ")
    parser.add_argument("--preflight-only", action="store_true", help="Select and warning-check frames without running detector inference")
    args = parser.parse_args()
    run(args.output_dir, disjoint=args.disjoint, preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
