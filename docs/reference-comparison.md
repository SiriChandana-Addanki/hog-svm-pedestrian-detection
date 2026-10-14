# Reference HOG implementation comparison

The supplied reference uses the same OpenCV default people SVM, `(4, 4)` win stride, `1.05` scale, and default zero hit threshold. It differs by resizing to width at most 400 pixels, using `(4, 4)` padding, drawing raw HOG candidates without NMS, and drawing boxes in resized-image coordinates.

## Controlled comparison

`scripts.compare_reference` runs four preprocessing/padding combinations and two additional hit thresholds on a still image or selected annotated video frames. It reports the detector input size, confidence range, raw and post-NMS counts, frame-level TP/FP/FN at IoU 0.5, preprocessing/HOG time, and writes comparable overlays. For resized variants, boxes are mapped back to the source image before NMS and matching.

The initial check used `examples/input/pedestrian.png` (1243×819):

| Variant | HOG input | Raw boxes | After NMS | HOG latency (ms) |
|---|---:|---:|---:|---:|
| Current settings, padding 8 | 1243×819 | 13 | 13 | 652.474 |
| Original size, padding 4 | 1243×819 | 13 | 13 | 627.371 |
| Width 400, padding 8 | 400×263 | 4 | 3 | 40.069 |
| Reference: width 400, padding 4 | 400×263 | 3 | 2 | 30.761 |

These are single-run timings, not a benchmark. The reference's reduced box count is not evidence of better localization: its mapped boxes span groups of pedestrians. The supplied image has no ground-truth boxes, so no precision/recall claim is made for it.

The annotated diagnostic subset used crosswalk frames 1, 100, 200, 300, 378; fourway frames 1, 641, 1281; and night frames 1, 283, 565. At IoU 0.5 the aggregate counts were:

| Variant | TP | FP | FN |
|---|---:|---:|---:|
| Current original-size, padding 8, hit threshold 0.0 | 3 | 32 | 8 |
| Original-size, padding 4, hit threshold 0.0 | 3 | 32 | 8 |
| Original-size, padding 8, hit threshold 0.5 | 3 | 10 | 8 |
| Original-size, padding 8, hit threshold 1.0 | 3 | 1 | 8 |
| Reference width 400, padding 4, hit threshold 0.0 | 1 | 3 | 10 |

This is a small diagnostic sample, not a dataset-level estimate. It supported `hit_threshold=1.0` as an operating point: it retained the same three matched detections while eliminating most low-confidence false positives on these frames. The full evaluations below show the dataset-level result for that setting.

## Full dataset results at hit threshold 1.0

Each included video was evaluated sequentially at every annotated frame. Matching remains one-to-one at IoU ≥ 0.5. Latency percentiles are detector wall-clock latency in milliseconds; process CPU and `cap.read()` decode latency are independent measurements. All calls remain in the raw results. The evaluator flags calls over 2×wall-clock p95 as outliers but does not omit them.

The latency values below are from a separately recorded earlier run and are run-condition-dependent. The latest verified full run, including per-frame mean detector wall latency, is documented in the repository [README](../README.md) and its artifacts under `results/reproducibility/full_20261014T122930477Z_2ece2fb9/`.

| Clip | Frames | TP | FP | FN | Precision | Recall | F1 | Wall median | Wall P95 | Wall P99 | Outliers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Crosswalk | 378 | 169 | 71 | 209 | 0.7042 | 0.4471 | 0.5469 | 3,211.121 | 5,299.441 | 6,469.235 | 0 / 378 |
| Fourway | 1,281 | 425 | 142 | 856 | 0.7496 | 0.3318 | 0.4600 | 3,398.992 | 4,662.867 | 5,866.237 | 5 / 1,281 |
| Night | 565 | 274 | 60 | 291 | 0.8204 | 0.4850 | 0.6096 | 3,546.670 | 5,337.880 | 7,418.991 | 4 / 565 |

Fourway's max wall-clock call was 2,350,789.503 ms on frame 145; that call used 17,296.875 ms of process CPU and its `cap.read()` took 7.790 ms. Night's max was 3,032,520.469 ms on frame 540; that call used 18,187.500 ms of process CPU and its `cap.read()` took 5.603 ms. These calls were timed inside `detectMultiScale()` and are wall-clock stalls consistent with process scheduling or suspension. The measurements do not demonstrate a specific OS-level cause. Decode time is separately measured and does not account for these detector-call durations. Crosswalk's max was 7,355.580 ms on frame 100 (19,671.875 ms process CPU, 11.335 ms decode). Per-frame raw measurements are in `results/final/crosswalk.csv`, `results/final/fourway.csv`, and `results/final/night.csv`.

## Engineering decisions

- **Keep original-resolution input.** Resizing to 400 pixels made crosswalk's selected pedestrians too small for the fixed HOG window: it produced no detections on those five frames. It also reduced recall on the combined subset. No resize is added, so production boxes remain in original image coordinates without scaling ambiguity.
- **Keep padding `(8, 8)`.** On the selected subset, changing to `(4, 4)` did not change TP, FP, or FN counts and barely changed detection counts. There is no evidence to adopt it.
- **Set hit threshold to `1.0`.** The sampled confidence distribution showed a clear reduction in low-score false positives with no additional misses in the diagnostic frames. Full per-dataset results are reported separately.
- **Keep NMS.** NMS removes overlapping duplicates, but most remaining false positives are spatially separate HOG responses. It is not a general false-positive filter.
- **Retain confidence weights and raw/final distinction.** HOG scores are preserved on each raw candidate; the existing shared NMS stage returns final candidates to image, video, and API flows.

Regenerate the comparison with `python -m scripts.compare_reference --input examples/input/pedestrian.png` or specify `--video`, `--annotations`, and selected 1-based `--frames`.
