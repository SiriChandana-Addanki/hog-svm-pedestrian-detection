# Scale 1.08 validation: disjoint diagnostic sample

Diagnostic subset only; not full-dataset performance. Each clip contributes 20 evenly distributed 1-based annotation rows (60 frames inferred under each setting). The same decoded frame is passed to both configurations.

Only `scale` changes: 1.05 vs 1.08. `hit_threshold=1.0`, stride (4,4), padding (8,8), NMS IoU 0.3, and evaluator one-to-one confidence-ordered IoU >= 0.5 matching remain fixed.

The frame selector excludes every frame from the original 15-frame sample and the preceding 60-frame sample. Known anomalous Fourway row 1 is also outside the selected set, so no annotation was excluded from this primary comparison. All selected warning-flagged annotations remain included.

Detector-call wall totals cover all 20 selected frames per clip (60 calls per setting). Per-clip experiment elapsed also includes NMS and matching; shared decode is reported separately.

## Aggregate metrics

TP/FP/FN | Precision | Recall | F1

| Clip | Scale | TP/FP/FN | Precision | Recall | F1 | Raw detections | Post-NMS | Detector wall ms | Experiment s | Shared decode s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| crosswalk | 1.05 | 8/4/12 | 0.6667 | 0.4000 | 0.5000 | 13 | 12 | 61199.3 | 61.20 | 2.459 |
| crosswalk | 1.08 | 8/3/12 | 0.7273 | 0.4000 | 0.5161 | 11 | 11 | 39134.6 | 39.14 | 2.459 |
| fourway | 1.05 | 6/2/14 | 0.7500 | 0.3000 | 0.4286 | 9 | 8 | 73204.9 | 73.21 | 8.687 |
| fourway | 1.08 | 6/2/14 | 0.7500 | 0.3000 | 0.4286 | 9 | 8 | 43913.9 | 43.92 | 8.687 |
| night | 1.05 | 10/3/10 | 0.7692 | 0.5000 | 0.6061 | 13 | 13 | 157460.2 | 157.46 | 3.759 |
| night | 1.08 | 7/3/13 | 0.7000 | 0.3500 | 0.4667 | 10 | 10 | 39761.4 | 39.76 | 3.759 |
| COMBINED | 1.05 | 24/9/36 | 0.7273 | 0.4000 | 0.5161 | 35 | 33 | 291864.5 | 291.88 | 14.904 |
| COMBINED | 1.08 | 21/8/39 | 0.7241 | 0.3500 | 0.4719 | 30 | 29 | 122809.9 | 122.82 | 14.904 |

## Annotation preflight

- **crosswalk:** frames `2, 22, 40, 60, 81, 101, 121, 139, 159, 179, 200, 220, 240, 258, 278, 298, 319, 339, 357, 377`.
- **fourway:** frames `2, 69, 135, 202, 271, 339, 406, 472, 539, 606, 676, 743, 810, 876, 943, 1011, 1080, 1147, 1213, 1280`.
- **night:** frames `2, 30, 61, 89, 119, 150, 180, 208, 239, 267, 299, 327, 358, 386, 416, 447, 477, 505, 536, 564`.
- Warning-flagged selected annotations: 5. All selected flagged rows remain included; any explicit exclusion is listed above.
  - `dataset/crosswalk.csv: annotation row/frame 377, XYXY box=(-157, 420, 64, 862): coordinates extend outside the frame. The annotation is retained unchanged.`
  - `dataset/fourway.csv: annotation row/frame 2, XYXY box=(1115, -96, 2537, 2749): coordinates extend outside the frame; extreme size (1422x2845, 195.1% of frame area; warning threshold is >10%). The annotation is retained unchanged.`
  - `dataset/fourway.csv: annotation row/frame 1280, XYXY box=(1001, -4, 1737, 1468): coordinates extend outside the frame; extreme size (736x1472, 52.2% of frame area; warning threshold is >10%). The annotation is retained unchanged.`
  - `dataset/night.csv: annotation row/frame 536, XYXY box=(856, 412, 1229, 1158): coordinates extend outside the frame; extreme size (373x746, 13.4% of frame area; warning threshold is >10%). The annotation is retained unchanged.`
  - `dataset/night.csv: annotation row/frame 564, XYXY box=(1502, -227, 3061, 2892): coordinates extend outside the frame; extreme size (1559x3119, 234.5% of frame area; warning threshold is >10%). The annotation is retained unchanged.`

## Per-frame comparison

Rows list both scales as `TP/FP/FN`, raw->post-NMS, and matched IoUs. FP means unmatched by the evaluator at IoU >= 0.5; visual FP examples can still depict a pedestrian whose predicted box is poorly localized. No selected annotation rows were excluded.

| Clip | Frame | GT | Scale 1.05 TP/FP/FN | raw->post | matched IoU | Scale 1.08 TP/FP/FN | raw->post | matched IoU | Change |
|---|---:|---:|---|---:|---|---|---:|---|---|
| crosswalk | 2 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 22 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 40 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 60 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 81 | 1 | 0/1/1 | 1->1 | - | 0/0/1 | 0->0 | - | changed |
| crosswalk | 101 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 121 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 139 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| crosswalk | 159 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| crosswalk | 179 | 1 | 1/0/0 | 1->1 | 0.617 | 1/0/0 | 1->1 | 0.635 | same counts |
| crosswalk | 200 | 1 | 1/0/0 | 1->1 | 0.512 | 1/0/0 | 1->1 | 0.516 | same counts |
| crosswalk | 220 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| crosswalk | 240 | 1 | 1/0/0 | 1->1 | 0.666 | 1/0/0 | 1->1 | 0.682 | same counts |
| crosswalk | 258 | 1 | 1/0/0 | 1->1 | 0.596 | 1/0/0 | 1->1 | 0.590 | same counts |
| crosswalk | 278 | 1 | 1/0/0 | 1->1 | 0.580 | 1/0/0 | 1->1 | 0.589 | same counts |
| crosswalk | 298 | 1 | 1/0/0 | 1->1 | 0.683 | 1/0/0 | 1->1 | 0.694 | same counts |
| crosswalk | 319 | 1 | 1/0/0 | 1->1 | 0.732 | 1/0/0 | 1->1 | 0.714 | same counts |
| crosswalk | 339 | 1 | 1/0/0 | 2->1 | 0.593 | 1/0/0 | 1->1 | 0.536 | same counts |
| crosswalk | 357 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| crosswalk | 377 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 2 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 69 | 1 | 1/0/0 | 1->1 | 0.632 | 1/0/0 | 1->1 | 0.627 | same counts |
| fourway | 135 | 1 | 1/0/0 | 1->1 | 0.517 | 1/0/0 | 1->1 | 0.523 | same counts |
| fourway | 202 | 1 | 1/0/0 | 1->1 | 0.539 | 1/0/0 | 1->1 | 0.554 | same counts |
| fourway | 271 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 339 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 406 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 472 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| fourway | 539 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 606 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 676 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 743 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 810 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 876 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 943 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| fourway | 1011 | 1 | 1/0/0 | 1->1 | 0.623 | 1/0/0 | 1->1 | 0.574 | same counts |
| fourway | 1080 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| fourway | 1147 | 1 | 1/0/0 | 2->1 | 0.638 | 1/0/0 | 2->1 | 0.651 | same counts |
| fourway | 1213 | 1 | 1/0/0 | 1->1 | 0.688 | 1/0/0 | 1->1 | 0.688 | same counts |
| fourway | 1280 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 2 | 1 | 1/0/0 | 1->1 | 0.642 | 1/0/0 | 1->1 | 0.628 | same counts |
| night | 30 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| night | 61 | 1 | 1/0/0 | 1->1 | 0.524 | 1/0/0 | 1->1 | 0.522 | same counts |
| night | 89 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 119 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 150 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 180 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 208 | 1 | 1/0/0 | 1->1 | 0.513 | 0/0/1 | 0->0 | - | changed |
| night | 239 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 267 | 1 | 1/1/0 | 2->2 | 0.694 | 0/1/1 | 1->1 | - | changed |
| night | 299 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 327 | 1 | 1/0/0 | 1->1 | 0.673 | 1/0/0 | 1->1 | 0.660 | same counts |
| night | 358 | 1 | 1/0/0 | 1->1 | 0.600 | 1/0/0 | 1->1 | 0.613 | same counts |
| night | 386 | 1 | 1/0/0 | 1->1 | 0.608 | 1/0/0 | 1->1 | 0.626 | same counts |
| night | 416 | 1 | 1/0/0 | 1->1 | 0.505 | 1/0/0 | 1->1 | 0.515 | same counts |
| night | 447 | 1 | 1/0/0 | 1->1 | 0.583 | 0/0/1 | 0->0 | - | changed |
| night | 477 | 1 | 1/0/0 | 1->1 | 0.812 | 1/0/0 | 1->1 | 0.804 | same counts |
| night | 505 | 1 | 0/1/1 | 1->1 | - | 0/1/1 | 1->1 | - | same counts |
| night | 536 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |
| night | 564 | 1 | 0/0/1 | 0->0 | - | 0/0/1 | 0->0 | - | same counts |

## Changed-frame best-overlap details

For each frame where TP/FP/FN counts changed, this is the maximum IoU of any post-NMS prediction with the frame's ground truth, including predictions below the 0.5 match cutoff.

| Clip | Frame | Baseline best prediction IoU | Candidate best prediction IoU | Outcome |
|---|---:|---:|---:|---|
| crosswalk | 81 | 0.000 | 0.000 | FP count changed |
| night | 208 | 0.513 | 0.000 | lost TP |
| night | 267 | 0.694 | 0.000 | lost TP |
| night | 447 | 0.583 | 0.000 | lost TP |

A recovered match means a candidate-scale match on ground truth that is unmatched at baseline; a lost match is the reverse. Review the paired overlays in this directory before drawing localization conclusions.

## Visual overlays

Paired overlays are saved only for frames where the matched-GT/FP/FN outcome signature differs; frame numbers are in the file names.
