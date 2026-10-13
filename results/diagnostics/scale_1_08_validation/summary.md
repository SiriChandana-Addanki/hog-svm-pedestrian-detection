# Scale 1.08 validation: stratified diagnostic sample

Diagnostic subset only; not full-dataset performance. Each clip contributes 20 evenly distributed 1-based annotation rows (60 frames inferred under each setting). The same decoded frame is passed to both configurations.

Only `scale` changes: 1.05 vs 1.08. `hit_threshold=1.0`, stride (4,4), padding (8,8), NMS IoU 0.3, and evaluator one-to-one confidence-ordered IoU >= 0.5 matching remain fixed.

Metrics exclude only the known anomalous Fourway annotation row/frame 1. That frame was still inferred under both settings and its unchanged annotation comparison is reported separately. No other selected annotations are omitted. Raw/post-NMS counts in primary totals exclude that frame to keep denominators aligned.

Detector-call wall totals align with primary metrics: 20 Crosswalk + 19 Fourway + 20 Night calls; Fourway row 1 was inferred but its detector-call timing is omitted from primary totals with the known annotation anomaly. The per-clip experiment elapsed is a broader stopwatch that includes all 20 frames, including that row, plus NMS and matching; shared decode is reported separately. Combined detector-call wall time therefore covers 59 scored frames, while combined experiment elapsed covers all 60 inference frames.

## Aggregate metrics

TP/FP/FN | Precision | Recall | F1

| Clip | Scale | TP/FP/FN | Precision | Recall | F1 | Raw detections | Post-NMS | Detector wall ms | Experiment s | Shared decode s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| crosswalk | 1.05 | 8/3/12 | 0.7273 | 0.4000 | 0.5161 | 11 | 11 | 68035.0 | 68.04 | 2.242 |
| crosswalk | 1.08 | 8/3/12 | 0.7273 | 0.4000 | 0.5161 | 11 | 11 | 42425.5 | 42.43 | 2.242 |
| fourway | 1.05 | 5/3/14 | 0.6250 | 0.2632 | 0.3704 | 9 | 8 | 66988.4 | 71.17 | 8.129 |
| fourway | 1.08 | 6/2/13 | 0.7500 | 0.3158 | 0.4444 | 9 | 8 | 39434.9 | 42.91 | 8.129 |
| night | 1.05 | 10/1/10 | 0.9091 | 0.5000 | 0.6452 | 11 | 11 | 71991.7 | 72.00 | 3.491 |
| night | 1.08 | 9/2/11 | 0.8182 | 0.4500 | 0.5806 | 11 | 11 | 44407.3 | 44.41 | 3.491 |
| COMBINED | 1.05 | 23/7/36 | 0.7667 | 0.3898 | 0.5169 | 31 | 30 | 207015.1 | 211.20 | 13.862 |
| COMBINED | 1.08 | 23/7/36 | 0.7667 | 0.3898 | 0.5169 | 31 | 30 | 126267.7 | 129.75 | 13.862 |

## Known anomalous Fourway row 1 (separate, not in primary totals)

| Scale | TP | FP | FN | Raw | Post-NMS |
|---:|---:|---:|---:|---:|---:|
| 1.05 | 0 | 1 | 1 | 1 | 1 |
| 1.08 | 0 | 0 | 1 | 0 | 0 |

## Annotation preflight

- **crosswalk:** frames `1, 21, 41, 61, 80, 100, 120, 140, 160, 180, 199, 219, 239, 259, 279, 299, 318, 338, 358, 378`.
- **fourway:** frames `1, 68, 136, 203, 270, 338, 405, 473, 540, 607, 675, 742, 809, 877, 944, 1012, 1079, 1146, 1214, 1281`.
- **night:** frames `1, 31, 60, 90, 120, 149, 179, 209, 238, 268, 298, 328, 357, 387, 417, 446, 476, 506, 535, 565`.
- Warning-flagged selected annotations: 5. They remain included except the explicitly excluded Fourway row 1.
  - `dataset/crosswalk.csv: annotation row/frame 378, XYXY box=(-165, 415, 73, 891): coordinates extend outside the frame. The annotation is retained unchanged.`
  - `dataset/fourway.csv: annotation row/frame 1, XYXY box=(631, -375, 3142, 4647): coordinates extend outside the frame; extreme size (2511x5022, 608.1% of frame area; warning threshold is >10%). The annotation is retained unchanged.`
  - `dataset/fourway.csv: annotation row/frame 1281, XYXY box=(1010, -23, 1782, 1522): coordinates extend outside the frame; extreme size (772x1545, 57.5% of frame area; warning threshold is >10%). The annotation is retained unchanged.`
  - `dataset/night.csv: annotation row/frame 535, XYXY box=(834, 426, 1203, 1165): coordinates extend outside the frame; extreme size (369x739, 13.2% of frame area; warning threshold is >10%). The annotation is retained unchanged.`
  - `dataset/night.csv: annotation row/frame 565, XYXY box=(1469, -209, 2998, 2850): coordinates extend outside the frame; extreme size (1529x3059, 225.6% of frame area; warning threshold is >10%). The annotation is retained unchanged.`

## Per-frame comparison

Rows list both scales as `TP/FP/FN`, raw->post-NMS, and matched IoUs. FP means unmatched by the evaluator at IoU >= 0.5; visual FP examples can still depict a pedestrian whose predicted box is poorly localized. Anomaly row 1 remains shown separately and excluded from primary analysis.

| Clip | Frame | GT | Scale 1.05 TP/FP/FN | raw->post | matched IoU | Scale 1.08 TP/FP/FN | raw->post | matched IoU | Change |
|---|---:|---:|---|---:|---|---|---:|---|---|
| crosswalk | 1 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 21 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 41 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 61 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 80 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 100 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 120 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 140 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| crosswalk | 160 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| crosswalk | 180 | 1 | 1/0/0 | 1->1 | 0.600 | 1/0/0 | 1->1 | 0.596 | same counts |
| crosswalk | 199 | 1 | 1/0/0 | 1->1 | 0.523 | 1/0/0 | 1->1 | 0.542 | same counts |
| crosswalk | 219 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| crosswalk | 239 | 1 | 1/0/0 | 1->1 | 0.722 | 1/0/0 | 1->1 | 0.710 | same counts |
| crosswalk | 259 | 1 | 1/0/0 | 1->1 | 0.637 | 1/0/0 | 1->1 | 0.641 | same counts |
| crosswalk | 279 | 1 | 1/0/0 | 1->1 | 0.598 | 1/0/0 | 1->1 | 0.590 | same counts |
| crosswalk | 299 | 1 | 1/0/0 | 1->1 | 0.691 | 1/0/0 | 1->1 | 0.702 | same counts |
| crosswalk | 318 | 1 | 1/0/0 | 1->1 | 0.706 | 1/0/0 | 1->1 | 0.693 | same counts |
| crosswalk | 338 | 1 | 1/0/0 | 1->1 | 0.516 | 1/0/0 | 1->1 | 0.506 | same counts |
| crosswalk | 358 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 378 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 1 | 1 | 0/1/1 | 1->1 | - | 0/0/1 | 0->0 | - | excluded anomaly |
| fourway | 68 | 1 | 1/0/0 | 1->1 | 0.625 | 1/0/0 | 1->1 | 0.627 | same counts |
| fourway | 136 | 1 | 0/1/1 | 1->1 | - | 1/0/0 | 1->1 | 0.509 | changed |
| fourway | 203 | 1 | 1/0/0 | 1->1 | 0.539 | 1/0/0 | 1->1 | 0.521 | same counts |
| fourway | 270 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 338 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 405 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 473 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| fourway | 540 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 607 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 675 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 742 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 809 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 877 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 944 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 1012 | 1 | 1/0/0 | 1->1 | 0.598 | 1/0/0 | 1->1 | 0.577 | same counts |
| fourway | 1079 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| fourway | 1146 | 1 | 1/0/0 | 2->1 | 0.595 | 1/0/0 | 2->1 | 0.583 | same counts |
| fourway | 1214 | 1 | 1/0/0 | 1->1 | 0.736 | 1/0/0 | 1->1 | 0.741 | same counts |
| fourway | 1281 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 1 | 1 | 1/0/0 | 1->1 | 0.710 | 1/0/0 | 1->1 | 0.651 | same counts |
| night | 31 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| night | 60 | 1 | 1/0/0 | 1->1 | 0.579 | 1/0/0 | 1->1 | 0.591 | same counts |
| night | 90 | 1 | 1/0/0 | 1->1 | 0.630 | 1/0/0 | 1->1 | 0.610 | same counts |
| night | 120 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 149 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 179 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 209 | 1 | 1/0/0 | 1->1 | 0.600 | 1/0/0 | 1->1 | 0.620 | same counts |
| night | 238 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 268 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 298 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 328 | 1 | 1/0/0 | 1->1 | 0.617 | 1/0/0 | 1->1 | 0.621 | same counts |
| night | 357 | 1 | 1/0/0 | 1->1 | 0.559 | 1/0/0 | 1->1 | 0.544 | same counts |
| night | 387 | 1 | 1/0/0 | 1->1 | 0.601 | 1/0/0 | 1->1 | 0.603 | same counts |
| night | 417 | 1 | 1/0/0 | 1->1 | 0.507 | 1/0/0 | 1->1 | 0.500 | same counts |
| night | 446 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 476 | 1 | 1/0/0 | 1->1 | 0.777 | 1/0/0 | 1->1 | 0.771 | same counts |
| night | 506 | 1 | 1/0/0 | 1->1 | 0.522 | 0/1/1 | 1->1 | - | changed |
| night | 535 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 565 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |

## Changed-frame best-overlap details

For each frame where TP/FP/FN counts changed, this is the maximum IoU of any post-NMS prediction with the frame ground truth, including predictions below the 0.5 match cutoff.

| Clip | Frame | Baseline best prediction IoU | Candidate best prediction IoU | Outcome |
|---|---:|---:|---:|---|
| fourway | 136 | 0.497 | 0.509 | recovered TP |
| night | 506 | 0.522 | 0.476 | lost TP |

The Fourway frame 136 recovery is marginal (candidate IoU just over 0.5). At Night frame 506 both configurations produce a box, but the candidate box falls below the matching threshold; this is a localization/matching loss, not an absent detection. Review the paired overlays before drawing wider conclusions.

## Visual overlays

Paired baseline/candidate overlays are saved under the clip subdirectories; filenames identify recovered, lost, or false-positive examples when such cases exist in that clip.
