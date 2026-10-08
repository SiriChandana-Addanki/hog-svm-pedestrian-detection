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

The video evaluator reads frames sequentially, because some codecs report an unreliable frame count. CSV annotations are one-based by row and use pixel `x,y,w,h`; boxes are compared at IoU ≥ 0.5 using one-to-one confidence-ordered matching. It reports detector wall-clock latency, detector process CPU time summed across process threads, and `cap.read()` video decode latency as separate distributions. Process CPU time can exceed wall time when work runs across multiple threads. Each distribution includes mean, median, p95, p99, minimum, and maximum; percentiles use the nearest-rank definition. Detector wall-time outliers are counted using `latency > 2 × detector wall-time p95`; every sample remains included in all statistics. Use `--timings-csv path.csv` to retain per-evaluated-frame raw measurements. Long calls are wall-clock stalls consistent with process scheduling or suspension; the evaluator does not identify their OS-level cause. A limited run is not a full-dataset result.

Run the complete crosswalk evaluation:

```powershell
python -m scripts.evaluate_video --video dataset/crosswalk.avi --annotations dataset/crosswalk.csv
```

Append `--timings-csv results/crosswalk_latency.csv` to save per-evaluated-frame raw timing samples.

Benchmark one image with a separate first-call (cold) latency and warm-call distribution:

```powershell
python -m scripts.benchmark --input examples/input/pedestrian.png --runs 20
```

## Measured results

Use the median, p95, and p99 as the primary summaries of detector wall-clock latency; wall-clock stalls can make the mean and maximum unrepresentative. The evaluator always prints maximum and outlier counts for transparency. Those values are raw measurements, not estimates of steady-state inference speed. Results depend on the machine and run conditions; rerun the commands above for a local measurement.

Full evaluation on the included clips (detector wall-clock latency, milliseconds):

| Clip | Evaluated frames | Median | P95 | P99 | Calls > 2×P95 |
|---|---:|---:|---:|---:|---:|
| Crosswalk | 378 | 3,777.595 | 4,321.959 | 5,036.236 | 2 |
| Fourway | 1,281 | 3,223.860 | 4,293.305 | 5,577.873 | 5 |
| Night | 565 | 3,306.285 | 4,507.020 | 51,131.841 | 21 |

The included crosswalk, fourway, and night clips have separate scene characteristics. Do not generalize one clip's detection metrics to the others.

## Limitations

The pretrained detector has a fixed HOG window and can miss small or distant pedestrians. Results depend on image scale, scene, and hardware. API end-to-end latency and container behavior must be measured separately from detector inference.
