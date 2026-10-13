# HOG + SVM Pedestrian Detection

Classical pedestrian detection using OpenCV's pretrained HOG + SVM detector. The project adds configurable inference, non-maximum suppression, dataset evaluation, latency benchmarking, a FastAPI endpoint, tests, Docker, and GitHub Actions CI.

## Architecture

Image → HOG + SVM → Raw Detections → NMS → Final Detections → Metrics / API

## Setup

Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .hog
.\.hog\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Detect an image

```powershell
python -m scripts.detect_image --input examples/input/pedestrian.png --output examples/output/detections.png
```

### Run the API locally

```powershell
python -m uvicorn api.main:app --reload
```

- Health check: `GET http://localhost:8000/health`
- Image detection: `POST http://localhost:8000/detect` with an image uploaded as a multipart form field named `file`.

### Run tests

```powershell
python -m pytest -q
```

### Run with Docker

```powershell
docker compose up --build -d
docker compose ps
```

Check the service:

```powershell
curl.exe http://localhost:8000/health
```

Stop the container:

```powershell
docker compose down
```

The `/detect` endpoint accepts an image upload and returns image dimensions, a request ID, raw and final detections, the number of suppressed detections, and detector latency.

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

Full evaluation on the included clips with the current `hit_threshold=1.0` (latency in ms; IoU threshold 0.5):

| Clip | Frames | TP | FP | FN | Precision | Recall | F1 | Wall median | Wall P95 | Wall P99 | Calls > 2×P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Crosswalk | 378 | 169 | 71 | 209 | 0.7042 | 0.4471 | 0.5469 | 3,211.121 | 5,299.441 | 6,469.235 | 0 |
| Fourway | 1,281 | 425 | 142 | 856 | 0.7496 | 0.3318 | 0.4600 | 3,398.992 | 4,662.867 | 5,866.237 | 5 |
| Night | 565 | 274 | 60 | 291 | 0.8204 | 0.4850 | 0.6096 | 3,546.670 | 5,337.880 | 7,418.991 | 4 |

Latency percentiles summarize detector wall-clock time; they exclude decode time because decoding is measured separately. The evaluator still prints and retains each distribution's raw maximum, but maximum values are not representative inference latency. The included clips have separate scene characteristics; do not generalize one clip's detection metrics to the others. These measurements are machine- and run-condition-dependent.

## Limitations

The pretrained detector has a fixed HOG window and can miss small or distant pedestrians. Results depend on image scale, scene, and hardware. API end-to-end latency and container behavior must be measured separately from detector inference.
