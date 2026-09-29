# Netra-One Pipeline Optimization — Intern Task

A four-stream 1080p video analytics reference implementation focused on asynchronous ingestion, latest-frame-wins backpressure, batched ONNX Runtime inference, bounded secondary-model scheduling, GPU memory monitoring, and reproducible benchmarking.

## Repository contents

```text
netra-one-intern-task/
├── README.md
├── ANSWERS.md
├── requirements.txt
├── export_models.py
├── baseline_naive.py
├── optimized_pipeline.py
├── benchmark.py
├── smoke_test.py
├── yolo11n.onnx
├── yolo11n-pose.onnx
├── assets/
│   ├── sample_1080p.mp4
│   └── test.mp4
└── results/
    ├── benchmark_report.md
    └── benchmark_report.json
```

Include the ONNX models, video assets, and generated result files only when their size and redistribution terms are appropriate for the assessment. If the clips or model weights cannot be redistributed, document how to obtain or generate them instead.

## Setup and run (Windows PowerShell)

Run commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Export the default detector and pose model to ONNX
python export_models.py

# Run smoke tests
python smoke_test.py

# Benchmark using the rectangular detector input (384x640)
python benchmark.py --video "assets\sample_1080p.mp4" --streams 4 --duration 30

# Benchmark using the square detector input (640x640), matching the required input shape
python benchmark.py --video "assets\test.mp4" --streams 4 --duration 30 --det-shape square

# Inspect available options
python benchmark.py --help
```

Optional diagnostics:

```powershell
# Fault-injection and crowd-surge tests
python benchmark.py --video "assets\sample_1080p.mp4" --streams 4 --duration 30 --extras

# Optional ablation using variable batch buckets
python benchmark.py --video "assets\test.mp4" --streams 4 --duration 30 --dynamic-batch
```

Use `--extras` for diagnostic scenarios, not for the primary real-time comparison. Keep hardware, video, stream count, duration, and options identical when comparing runs. The benchmark writes `results\benchmark_report.md` and `results\benchmark_report.json`; each run overwrites these paths, so copy the reports to separate filenames if you want to preserve multiple configurations.

The export script creates `yolo11n.onnx` and `yolo11n-pose.onnx` and reports FP16/FP32 initializer counts. FP16 weight storage does **not** mean that runtime inputs or activations use FP16. The runtime input dtype is reported separately. The benchmark runs the naive baseline, real-time optimized pipeline, and saturation test in separate processes, performs a 30-iteration warmup, measures for the requested duration, and synchronizes CUDA before stopping each tick timer. The default provider order is ONNX Runtime CUDA Execution Provider followed by CPU fallback. Use `--tensorrt` only if the ONNX Runtime TensorRT Execution Provider and its dependencies are installed.

## Hardware used for the latest recorded run

- GPU: NVIDIA GeForce RTX 2050, 4096 MiB
- Driver: 560.70
- OS: Windows (PowerShell)
- CPU: 12 logical CPUs reported by Python
- Exact CPU model and system RAM were not recorded

## Architecture

1. **Asynchronous ingestion.** One `StreamReader` thread per stream feeds a bounded depth-1 `LatestFrameQueue`. A new frame replaces an unread stale frame. File EOF rewinds; open failures retry; stream faults are isolated.
2. **Micro-batch collection.** The production collector uses `BATCH_TIMEOUT_MS = 2.0` after the first frame arrives. `IDLE_WAIT_MS = 5.0` controls idle waiting when no frame is available. Silent streams are excluded after the liveness window. The smoke test's idle-timeout test has its own timeout; its printed value should not be confused with the production batch-window constant.
3. **Preprocessing.** Reader threads letterbox frames into uint8 frame buffers. The inference worker copies them into reusable pinned uint8 staging buffers. On the GPU path, channel reorder, NCHW conversion, dtype conversion, and normalization occur on device. ONNX Runtime I/O binding reuses preallocated GPU input/output buffers.
4. **Primary detector.** Model A runs as a fixed-shape batch across active streams. Detection candidate limits are applied per image; NMS groups are independent by image and class. The CLI supports rectangular `384x640` and square `640x640` input configurations.
5. **Pose stage.** The scheduler sends person detections only to Model B, applies confidence/area thresholds and a per-stream cooldown, and caps the batch at four ROIs per tick. Pose output is decoded into 17 keypoints and mapped back to original-frame coordinates. Only the latest tick's pose results are retained in memory; a cumulative count is reported.
6. **Memory guardrails.** PyTorch's per-process fraction and per-session ONNX Runtime CUDA arena budgets are configured. The benchmark measures process VRAM where supported or labels device-wide telemetry when per-process usage is unavailable. These controls reduce risk but are not a global OS-level VRAM reservation.
7. **Baseline.** `baseline_naive.py` uses sequential, synchronous, batch-size-1 inference as a comparison reference.

## Latest benchmark result — square detector input

Date: 2026-09-29

Command:

```powershell
python benchmark.py --video "assets\test.mp4" --streams 4 --duration 30 --det-shape square
```

Four readers loop the same local 1080p clip. Live software decoding is included. This is a local-file test, not four independent camera network connections. The detector input is `640x640`, and the runtime input dtype is `float32`; FP16 ONNX weight storage is reported separately.

| Metric | Naive baseline | Optimized real-time pipeline | Target / interpretation |
|---|---:|---:|---|
| Aggregate throughput | 10.8 FPS | 119.0 FPS | Target ≥120 FPS; not met |
| Effective per-stream throughput | 2.7 FPS | 29.8 FPS | Target ≥30 FPS; not met |
| Mean batch-loop latency | 370.0 ms | 27.7 ms | Optimized mean is lower |
| P50 / P95 / P99 latency | 363.3 / 441.1 / 510.2 ms | 27.2 / 31.1 / 43.7 ms | P99 target ≤33.3 ms; not met |
| Peak device-wide VRAM | 751 MB | 954 MB | Target ≤1536 MB; met |
| Host CPU utilization | 7.7% | 14.8% | Measured |
| Dropped frames | N/A; baseline lags | 30 | 99.2% of 120.0 offered FPS served |
| Frame age P50 / P99 / max | Not reported | 39.8 / 65.1 / 83.1 ms | Freshness telemetry |

The VRAM values are device-wide readings and include memory used by other applications. At this run, approximately 532 MB was already in use by other applications; the report estimates about 422 MB net of that background usage for the optimized run. This subtraction is approximate, not a direct process-only measurement.

### Saturation capacity

Saturation mode removes real-time reader pacing and is a capacity diagnostic, not the primary real-time target measurement.

| Metric | Naive baseline | Optimized saturated pipeline |
|---|---:|---:|
| Aggregate throughput | 10.8 FPS | 62.5 FPS |
| P99 batch-loop latency | 510.2 ms | 127.7 ms |
| Average batch size | 1 | 4.00 |

### Per-stage mean latency (optimized real-time run)

| Stage | Mean |
|---|---:|
| `pre_a` | 2.53 ms |
| Detector | 18.70 ms |
| Detection decode | 3.90 ms |
| ROI scheduling | 0.09 ms |
| `pre_b` | 0.02 ms |
| Pose model | 0.55 ms |

Average batch size was 3.39. The run processed 63 Model B ROIs and decoded 57 pose outputs. No ROIs were deferred by the scheduler in this run.

### Rectangular-input comparison

A separate run with `assets\sample_1080p.mp4` and the default rectangular `384x640` detector input recorded 119.2 aggregate FPS, 29.8 FPS per stream, and 41.0 ms P99 latency. It also missed the aggregate throughput, per-stream throughput, and P99 latency targets. Since both the video content and detector shape differ from the square-input run above, these measurements should not be treated as a controlled comparison of detector shape alone.

## Smoke tests

Run:

```powershell
python smoke_test.py
```

The latest smoke-test run passed all nine tests, covering queue replacement, letterbox geometry, detection decoding, pose/keypoint decoding, ROI scheduling, multi-tick operation with a simulated dead stream, collector idle timeout, stream blackout/recovery, and preloaded-clip behavior. These are software/simulation tests; they do not replace real-model GPU performance validation or keypoint-accuracy evaluation.

## Measurement notes and limitations

- For the latest square-input run, VRAM was within the target, but aggregate throughput (119.0 vs. 120 FPS), per-stream throughput (29.8 vs. 30 FPS), and P99 latency (43.7 vs. 33.3 ms) did not meet the targets.
- The square run uses the task's literal detector input dimensions `640x640`. The benchmark reports runtime input dtype as `float32`; do not describe it as FP16-activation inference or INT8 inference.
- The test loops one local clip across four independent readers; it does not test four independent RTSP camera networks.
- Live software decode is included by default. `--preload` excludes video decoding from the optimized measurement and should not be used to claim live-decode performance.
- On Windows WDDM, per-process GPU memory may not be available through NVML. Device-wide readings include other applications.
- Latency and throughput depend on hardware, video content, execution providers, driver state, and configuration. Use the report generated by the exact final source and configuration submitted.
