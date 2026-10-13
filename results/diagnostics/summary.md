# Targeted detection diagnostics

Five frames per clip were selected at evenly spaced 1-based annotation rows.
Video frames were decoded sequentially; detector inference ran only on selected frames.
Ground truth uses the evaluator CSV row mapping; predictions use configured detector and NMS.
Matching uses evaluator IoU threshold 0.5 and confidence-ordered greedy one-to-one matching.
Ground-truth boxes are green; post-NMS predictions are red. Box clipping is for display only.

## crosswalk

Annotation rows: 378; sequentially decoded through row 378; decoded frame count equals annotation rows.

| Frame | TP | FP | FN | Image |
|---:|---:|---:|---:|---|
| 1 | 0 | 0 | 1 | `crosswalk/frame_000001.jpg` |
| 95 | 0 | 0 | 1 | `crosswalk/frame_000095.jpg` |
| 189 | 1 | 0 | 0 | `crosswalk/frame_000189.jpg` |
| 284 | 1 | 0 | 0 | `crosswalk/frame_000284.jpg` |
| 378 | 0 | 0 | 1 | `crosswalk/frame_000378.jpg` |

## fourway

Annotation rows: 1281; sequentially decoded through row 1281; decoded frame count equals annotation rows.

| Frame | TP | FP | FN | Image |
|---:|---:|---:|---:|---|
| 1 | 0 | 1 | 1 | `fourway/frame_000001.jpg` |
| 321 | 0 | 1 | 1 | `fourway/frame_000321.jpg` |
| 641 | 0 | 0 | 1 | `fourway/frame_000641.jpg` |
| 961 | 0 | 0 | 1 | `fourway/frame_000961.jpg` |
| 1281 | 0 | 0 | 1 | `fourway/frame_001281.jpg` |

## night

Annotation rows: 565; sequentially decoded through row 565; decoded frame count equals annotation rows.

| Frame | TP | FP | FN | Image |
|---:|---:|---:|---:|---|
| 1 | 1 | 0 | 0 | `night/frame_000001.jpg` |
| 142 | 0 | 0 | 1 | `night/frame_000142.jpg` |
| 283 | 0 | 0 | 1 | `night/frame_000283.jpg` |
| 424 | 0 | 1 | 1 | `night/frame_000424.jpg` |
| 565 | 0 | 0 | 1 | `night/frame_000565.jpg` |
