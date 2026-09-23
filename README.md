# HOG + SVM Pedestrian Detection

Production-oriented pedestrian detection system built around OpenCV's pretrained HOG + SVM people detector, with explicit post-processing, benchmarking, automated tests, FastAPI inference, Docker support, and CI.

## Overview

This project focuses on engineering a classical computer-vision detection pipeline rather than training a new model.

The system uses OpenCV's pretrained HOG + SVM pedestrian detector and builds an inference layer around it for:

- pedestrian detection
- non-maximum suppression (NMS)
- latency benchmarking
- structured detection metrics
- automated testing
- REST API inference
- containerized deployment
- CI validation

## Architecture

Input Image
     |
     v
Preprocessing
     |
     v
HOG + SVM Detector
     |
     v
Raw Detections
     |
     v
Non-Maximum Suppression
     |
     v
Final Detections
     |
     +----> Detection Metrics
     |
     +----> Latency Benchmark
     |
     +----> FastAPI
                  |
                  v
             JSON Response



# TECH STACK

Python 3.10
OpenCV 4.14
NumPy
FastAPI
Uvicorn
Pytest
Docker
GitHub Actions


# PROJECT STRUCTURE

hog-svm-pedestrian-detection/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── README.md
│
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   └── production-checklist.md
│
├── examples/
│   ├── input/
│   └── output/
│
├── scripts/
│   ├── benchmark.py
│   ├── detect_image.py
│   └── detect_video.py
│
├── src/
│   ├── config.py
│   ├── detector.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── postprocessing.py
│   └── preprocessing.py
│
├── tests/
│   ├── test_api.py
│   ├── test_detector.py
│   ├── test_evaluation.py
│   └── test_postprocessing.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements.lock.txt
├── metrics.json
└── LICENSE




# SETUP

Create and activate the virtual environment:
        python -m venv .hog

Windows:
        .hog\Scripts\activate


Install Dependencies:

python -m pip install -r requirements.txt


# Run Detection

python -m scripts.detect_image \
    --input examples/input/pedestrian.png \
    --output examples/output/baseline.png


The detector returns:

bounding boxes
detector confidence values
detection count
inference latency



Non-Maximum Suppression

NMS is implemented as a separate post-processing stage.

Run:

python -m scripts.detect_image \
    --input examples/input/pedestrian.png \
    --output examples/output/nms.png \
    --iou-threshold 0.3



The pipeline reports:

raw detections
final detections
suppressed detections
suppression ratio
inference latency


The current example image produced:

Raw detections:        13
Final detections:      13
Suppressed detections: 0
Suppression ratio:     0.0

These values are measurements from the current example image, not general model-performance claims.


Benchmarking

Run:   python -m scripts.benchmark \
    --input examples/input/pedestrian.png \
    --runs 20



Current benchmark:



| Metric                 |     Result |
| ---------------------- | ---------: |
| Image resolution       | 1243 × 819 |
| Runs                   |         20 |
| Mean latency           | 609.049 ms |
| Median latency         | 609.688 ms |
| P50 latency            | 610.799 ms |
| P95 latency            | 674.131 ms |
| Minimum latency        | 486.182 ms |
| Maximum latency        | 674.131 ms |
| Mean detections        |         13 |
| Detection count stable |        Yes |



These measurements were obtained on the development environment and should not be interpreted as hardware-independent performance.


API

Start the API:
python -m uvicorn api.main:app --reload

Health check:
GET /health

Detection endpoint:

POST /detect


Example using curl:

curl -X POST \
  -F "file=@examples/input/pedestrian.png" \
  http://127.0.0.1:8000/detect



The API returns structured detection information including:

image dimensions
raw detection count
final detection count
suppressed detection count
inference latency
bounding boxes
confidence values


Interactive API documentation is available through FastAPI at:

http://127.0.0.1:8000/docs


Testing

Run the complete test suite:

python -m pytest -q

current result :

6 passed

The tests cover:

detector initialization
detector output structure
NMS behavior
API health endpoint
image detection endpoint
invalid input handling



Docker

Build the image:

docker build -t hog-svm-pedestrian-detection .


Run the service:

docker compose up --build


CI

The repository includes a GitHub Actions workflow for automated validation.

The CI pipeline is intended to verify the project environment and test suite before changes are merged.

Evaluation

Detection quality metrics such as precision, recall, F1-score, and mAP require ground-truth annotations.

The current repository does not claim those metrics without an appropriate annotated evaluation dataset.

Operational measurements such as latency, detection counts, suppression behavior, and API behavior can be measured independently.

See:

docs/evaluation.md

for the evaluation approach.

Engineering Focus

The project is intentionally structured around an engineering workflow:



Baseline
   ↓
Observe
   ↓
Measure
   ↓
Post-processing
   ↓
Evaluate
   ↓
Benchmark
   ↓
Test
   ↓
API
   ↓
Containerize
   ↓
CI
   ↓
Document


The goal is to demonstrate how a classical computer-vision model can be wrapped in a reproducible, testable, measurable inference system.




# Limitations

The detector uses OpenCV's pretrained HOG + SVM people detector.
No model retraining is performed in this project.
Detection quality metrics require annotated ground-truth data.
Current benchmark results are environment-dependent.
HOG + SVM inference is CPU-oriented and can have substantially higher latency than modern optimized detection models.
The current example benchmark is based on a single input image and should not be treated as a representative dataset-level evaluation.


# Reproducibility


The repository includes:

pinned core runtime dependencies
a dependency lock file
deterministic project structure
automated tests
benchmark scripts
documented configuration
Docker support
CI configuration


License 

