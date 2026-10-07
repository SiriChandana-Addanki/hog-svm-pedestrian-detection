# HOG + SVM Pedestrian Detection

Classical pedestrian detection using OpenCV's pretrained HOG + SVM detector. The project adds configurable inference, non-maximum suppression, dataset evaluation, latency benchmarking, a FastAPI endpoint, tests, Docker, and GitHub Actions CI.

## Architecture

Image → HOG + SVM → Raw Detections → NMS → Final Detections → Metrics / API

## Setup

```powershell
python -m venv .hog
.hog\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Detect an image:

```powershell
python -m scripts.detect_image --input examples/input/pedestrian.png --output examples/output/detections.png
```

Run the API:

```powershell
python -m uvicorn api.main:app --reload
```

`GET /health` checks the service. `POST /detect` accepts an image upload and returns its dimensions, request ID, raw and final detections, suppressed count, and detector latency.

Run tests with `python -m pytest -q`. Build and run the container with `docker compose up --build`.

## Evaluation

The video evaluator reads frames sequentially, because some codecs report an unreliable frame count. CSV annotations are one-based by row and use pixel `x,y,w,h`; boxes are compared at IoU ≥ 0.5 using one-to-one confidence-ordered matching. Results report decoded, annotated, and evaluated frame counts, sampling interval, TP/FP/FN, precision/recall/F1, and detector inference latency percentiles. A limited run is not a full-dataset result.

Run the complete crosswalk evaluation:

```powershell
python -m scripts.evaluate_video --video dataset/crosswalk.avi --annotations dataset/crosswalk.csv
```

Benchmark one image with a separate first-call (cold) latency and warm-call distribution:

```powershell
python -m scripts.benchmark --input examples/input/pedestrian.png --runs 20
```

## Measured results

- **Full crosswalk evaluation:** command `python -m scripts.evaluate_video --video dataset/crosswalk.avi --annotations dataset/crosswalk.csv`; 378 frames decoded and evaluated, 378 CSV annotation rows, sampling interval 1, IoU threshold 0.5. TP=173, FP=979, FN=205; precision=0.1502, recall=0.4577, F1=0.2261. Detector-call latency, including the first call: mean 3267.411 ms, median 3193.679 ms, p95 3950.198 ms, min 1345.467 ms, max 4439.053 ms.
- **Single-image benchmark:** command `python -m scripts.benchmark --input examples/input/pedestrian.png --runs 10`; 1243 × 819 image, first detector-call latency 642.288 ms; 10 subsequent warm calls: mean 621.315 ms, median/p50 635.270 ms, p95 662.130 ms, min 491.316 ms, max 662.130 ms; mean detections 13 (stable in 10 runs). Development-machine measurement, not a throughput claim.

The first-frame visual check confirmed the CSV box surrounds the visible pedestrian; after correcting the evaluator's coordinate conversion, that frame still has no prediction at IoU 0.5 because the HOG candidates are substantially larger than the annotation. The full-dataset metrics show the detector also produces many false positives on this clip. The included `fourway` and `night` videos were not evaluated; do not generalize the crosswalk score to them.

## Limitations

The pretrained detector has a fixed HOG window and can miss small or distant pedestrians. Results depend on image scale, scene, and hardware. API end-to-end latency and container behavior must be measured separately from detector inference.
