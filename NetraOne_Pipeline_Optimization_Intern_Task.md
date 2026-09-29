# 0xAstra Private Limited — Technical Assessment
## Role: AI Systems & Pipeline Optimization Intern (Netra-One Engine)
**Time Allocation:** 24 Hours (Rapid Screening Sprint)  
**Target Platform:** Local NVIDIA GPU (4GB+ VRAM) or Free Google Colab (`T4` GPU)  
**Deliverables:** Code Repository + Automated `benchmark.py` Script + Written Engineering Answers (`ANSWERS.md`) *(No video recording required)*

---

## 1. Mission Context

At **0xAstra**, our core intelligence platform **Netra-One** processes high-density, multi-camera 1080p RTSP surveillance streams concurrently on a single tactical workstation GPU. In production, naive deep learning pipelines fail not because of model architecture, but because of **I/O blocking, CPU-to-GPU memory copy bottlenecks, unbatched single-frame execution (`Batch=1`), and unmanaged VRAM fragmentation** when multiple models co-exist on the same GPU.

Your objective in this **24-hour technical sprint** is to take a naive, bottlenecked 4-stream video analytics pipeline, re-architect it for **high-throughput asynchronous batched inference (`ONNX Runtime` / `TensorRT` FP16 or INT8)**, enforce a **strict VRAM budget with two-stage multi-model scheduling**, and prove your optimization gains using an automated benchmarking script.

---

## 2. The Problem: Naive Baseline (`baseline_naive.py`)

Below is the reference implementation of a broken, naive 4-stream pipeline. It exhibits five critical production anti-patterns:
1. **Sequential Decode Blocking:** Reads 4 streams sequentially in the same thread as GPU inference (`cv2.VideoCapture.read()`), stalling the GPU while the CPU decodes H.264 frames.
2. **Unbatched Inference (`Batch=1`):** Runs the primary detector 4 separate times per loop iteration in `FP32` instead of dynamically batching `[4, 3, 640, 640]`.
3. **Unscheduled Multi-Model Execution:** Runs a secondary model on every single detected bounding box synchronously (`B=1` crop loop) without ROI filtering, batching, or frame-skipping/scheduling.
4. **Redundant Host-to-Device Copies:** Performs resize, BGR-to-RGB conversion, and normalization sequentially in NumPy on the CPU before copying tensors to VRAM.
5. **Zero VRAM Guardrails:** Allocates unbounded GPU memory without pre-allocated buffers or CUDA memory ceiling enforcement.

```python
# baseline_naive.py — DO NOT USE THIS ARCHITECTURE IN PRODUCTION
import time
import cv2
import torch
from ultralytics import YOLO

def run_naive_pipeline(video_paths: list[str], num_steps: int = 300):
    # Anti-pattern 1: Unoptimized FP32 PyTorch weights loaded with default memory settings
    primary_detector = YOLO("yolo11n.pt").to("cuda")       # Stage 1: Person/Vehicle Detector
    secondary_model = YOLO("yolo11n-pose.pt").to("cuda")   # Stage 2: Secondary ROI Analyzer (Pose/Keypoints)

    caps = [cv2.VideoCapture(p) for p in video_paths]
    latencies_ms = []

    for step in range(num_steps):
        t0 = time.perf_counter()
        for stream_id, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            # Anti-pattern 2: Sequential B=1 inference per stream
            det_results = primary_detector(frame, verbose=False)

            # Anti-pattern 3: Unbatched synchronous crop loop for Secondary Model
            boxes = det_results[0].boxes.xyxy.cpu().numpy()
            for box in boxes[:5]:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
                if crop.size > 0:
                    _ = secondary_model(crop, verbose=False)

        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    for cap in caps:
        cap.release()
    return latencies_ms
```

---

## 3. Core Engineering Tasks

### Task 1: Asynchronous 4-Stream Ingestion & Batched Inference Pipeline
Build an optimized pipeline (`optimized_pipeline.py`) that ingests **4 concurrent 1080p (1920x1080 @ 30 FPS) video streams** (you may loop a single 1080p MP4 across 4 independent stream workers or simulate RTSP streams) and implements:
1. **Decoupled Asynchronous Stream Ingestion:**
   - Dedicated non-blocking reader threads/processes per stream.
   - **Bounded Ring Buffer / Latest-Frame Queue (`maxsize=1` or `2`):** Implement a strict **drop-oldest (latest-frame-wins)** backpressure policy so real-time streams never accumulate multi-second lag if inference momentarily slows down.
   - **Fault Resilience:** Gracefully handle simulated stream dropouts, corrupt/empty frames (`ret == False`), and automatic reconnection/rewind without crashing the pipeline or stalling healthy streams.
2. **Dynamic Frame Batching (`B <= 4`):**
   - Collect the latest available frames across active streams with a strict micro-batch timeout (e.g., `<= 5 ms` wait window) so a single slow stream never blocks the remaining 3 streams.
   - Vectorize preprocessing (letterbox resize, normalization, `NCHW` contiguous memory layout, pinned host memory `pin_memory=True` or direct GPU preprocessing).
3. **Model Export & Runtime Acceleration:**
   - Export both models from PyTorch `FP32` to **ONNX Runtime (`CUDAExecutionProvider` / `TensorRTExecutionProvider`)** or native **TensorRT (`FP16` or `INT8`)**.
   - Execute batched inference (`[B, 3, 640, 640]`) for the primary detector.

---

### Task 2: Strict VRAM Budget & Multi-Model Co-Location Scheduler
In production, Netra-One co-locates multiple specialized models inside a tight VRAM envelope. You must co-locate **two models concurrently** on the GPU under a **hard VRAM cap**:
* **Model A (Primary Detector):** `yolo11n` (or `yolov8n`) detecting `person` (class 0) and `car/truck` (classes 2, 7) across all 4 streams.
* **Model B (Secondary ROI Specialist):** `yolo11n-pose` (or `yolo11n-cls` / lightweight ONNX Face/ALPR backbone) triggered **only** on high-confidence detections (`conf >= 0.45`, minimum box area threshold) extracted from Model A.

**Mandatory VRAM & Scheduling Constraints:**
1. **Hard VRAM Ceiling (`<= 1536 MB` Peak GPU Memory):**
   - Enforce a hard programmatic memory cap at process startup (e.g., `torch.cuda.set_per_process_memory_fraction` and/or ONNX Runtime `gpu_mem_limit` arena configuration).
   - Your entire 4-stream pipeline (both models + preprocessing buffers + execution context) **must run in `< 1.5 GB` of peak allocated VRAM** without throwing `CUDA out of memory`.
2. **Smart ROI Batching & Scheduling:**
   - Instead of calling Model B in a `for` loop per crop (`B=1`), batch extracted ROIs across all 4 streams into a fixed/dynamic tensor batch (e.g., `max_rois_per_tick = 8`, sorted by priority/area) or implement an intelligent **stride/track-ID cooldown scheduler** (e.g., run Model B on a given bounding box region every $K$ frames rather than every single frame).
   - Document how your scheduler prevents a "crowd surge" (e.g., 40 people suddenly appearing across all 4 cameras) from blowing up frame latency (`P99`) or exhausting VRAM.

---

### Task 3: Automated Benchmark & Telemetry Script (`benchmark.py`)
Create a single-command CLI script (`python benchmark.py --streams 4 --duration 30`) that automatically runs **both** the `baseline_naive` pipeline and your `optimized_pipeline` over at least **300 iterations (or 30 seconds)** after a **30-iteration GPU warmup**, and outputs a structured Markdown & JSON comparison report.

Your `benchmark.py` **must capture and print the following exact table**:

| Metric | Naive Baseline (`FP32`, Sync `B=1`) | Optimized Pipeline (`FP16`/`INT8`, Async Batched) | Speedup / Delta |
| :--- | :---: | :---: | :---: |
| **Aggregate System Throughput (Total FPS)** | `e.g., 28.4 FPS` | `Target: >= 120.0 FPS (30 FPS x 4)` | `+X.XXx` |
| **Effective Per-Stream FPS (4 Streams)** | `e.g., 7.1 FPS` | `Target: >= 30.0 FPS` | `+X.XXx` |
| **Batch Loop Latency — Mean (`ms`)** | `ms` | `ms` | `-XX.X%` |
| **Batch Loop Latency — P50 / P95 / P99 (`ms`)** | `P50 / P95 / P99` | `Target P99 <= 33.3 ms` | `-XX.X%` |
| **Peak GPU VRAM Allocated / Reserved (`MB`)** | `MB` | `Must be <= 1536 MB` | `-XX.X%` |
| **Host CPU Utilization (`%`)** | `%` | `%` | `Delta` |
| **Dropped / Stale Frames Under Overload** | `N/A (Lags infinitely)` | `Count & Freshness Guarantee` | `Verified` |

> **Note on Timing Accuracy:** All GPU timings must use proper synchronization (`torch.cuda.synchronize()` or CUDA Events) around inference execution so asynchronous kernel launches do not produce fake sub-millisecond readings.

---

### Task 4: Written Technical Deep-Dive (`ANSWERS.md`)
Provide concise, rigorous engineering answers (bullet points + math/diagrams welcome, max 150–250 words per question) to the following **4 Netra-One systems questions**:

1. **Zero-Copy RTSP Ingestion at Scale (16 to 50 Cameras):**
   Why does `cv2.VideoCapture("rtsp://...")` collapse CPU utilization when scaling from 4 streams to 32 streams of 1080p H.264/H.265 video? Explain how a hardware-accelerated **GStreamer / DeepStream (`nvv4l2decoder` / `NVDEC` -> `NVMM` GPU memory)** or **PyNvVideoCodec** pipeline eliminates PCIe bandwidth bottlenecks between decoding and TensorRT inference.
2. **TensorRT Dynamic Batching vs. Fixed-Shape Engines:**
   When exporting an ONNX model with dynamic batch and ROI dimensions (`[B, 3, H, W]`), how do TensorRT **Optimization Profiles (`MIN`, `OPT`, `MAX`)** select CUDA kernels? What happens to inference latency if `MAX` is set too wide or if batch sizes fluctuate rapidly between `B=1` and `B=16`, and how would you stabilize kernel selection?
3. **INT8 Calibration & Small-Object Accuracy Degradation:**
   In outdoor perimeter surveillance, small distant targets (e.g., a `16x16` pixel drone or intruder at night) often suffer severe recall drops after naive Post-Training Quantization (`PTQ`) from `FP16` to `INT8`. Explain *why* activation clipping in `Concat`/`SiLU` or detection heads hurts small bounding box regression, and how you would configure **Entropy/MinMax calibration datasets** or **Mixed-Precision (`FP16` head + `INT8` backbone)** to preserve mAP.
4. **Sub-20ms Dynamic Model Hot-Swapping Under a Fixed 20GB VRAM Ceiling:**
   Suppose an RTX 4000 Ada workstation (`20 GB VRAM`) must run 6 always-on baseline surveillance models (`14 GB` used) and dynamically swap in 3 heavy incident-response models (`4.5 GB` each) only when an alert triggers—without dropping below 30 FPS on the primary streams. Describe your exact memory management architecture (e.g., pinned host RAM staging, pre-allocated CUDA memory pools, CUDA streams, and weight loading pitfalls) to achieve near-zero-stall model switching.

---

## 4. Submission Requirements & Repository Structure

Submit a GitHub repository link (or `.zip` archive) containing:

```text
netra-one-intern-task/
├── README.md                  # Exact setup commands, hardware used, and printed Benchmark Table
├── ANSWERS.md                 # Your responses to the 4 Systems Questions in Task 4
├── requirements.txt           # Reproducible dependencies
├── export_models.py           # Script to export YOLO11 weights to ONNX / TensorRT FP16/INT8
├── baseline_naive.py          # The unoptimized baseline for comparison
├── optimized_pipeline.py      # Your multi-stream async + VRAM-scheduled engine
└── benchmark.py               # Automated harness generating JSON + Markdown metrics
```

### Evaluation Criteria (100 Points)
| Category | Weight | What We Look For |
| :--- | :---: | :--- |
| **1. Pipeline Architecture & Concurrency** | **30 pts** | Lock-free/bounded latest-frame queues, micro-batch collector with timeout, zero blocking between RTSP/file readers and GPU worker, clean shutdown & error recovery. |
| **2. Inference & Preprocessing Speedup** | **25 pts** | Real `FP16`/`INT8` ONNX Runtime or TensorRT execution, vectorized/pinned memory preprocessing, accurate `torch.cuda.synchronize()` timing, and `P99 <= 33.3 ms`. |
| **3. VRAM Discipline & Multi-Model Scheduling** | **25 pts** | Strict `< 1536 MB` VRAM adherence, batched ROI execution or track/stride cooldown scheduling for Model B, immunity to crowd-surge OOMs. |
| **4. Systems Depth (`ANSWERS.md`)** | **20 pts** | First-principles understanding of NVDEC/PCIe transfers, TensorRT optimization profiles, INT8 quantization sensitivity, and pinned-memory CUDA stream staging. |

---
**0xAstra Private Limited — Engineering Recruitment**  
*Build fast. Measure honestly. Respect the hardware.*
