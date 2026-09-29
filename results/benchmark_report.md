# Netra-One Benchmark Report

- GPU: NVIDIA GeForce RTX 2050, 4096 MiB, 560.70
- Streams: 4 x 1080p | measured 30.0s per run after a 30-iteration in-place warmup
- Video: `assets\test.mp4` (real footage with people/vehicles, looped across 4 readers)
- Optimized model input dtype: float32 (FP16 weights are separately verified by export_models.py) | Model A input: 640x640 | Model B input: 256px | Shapes: fixed (Model A batch = streams, Model B batch = 4)
- ORT session provider lists (priority/fallback order): Model A=['CUDAExecutionProvider', 'CPUExecutionProvider']; Model B=['CUDAExecutionProvider', 'CPUExecutionProvider']
- Configured ORT CUDA arena budgets (MB): {'detector': 640, 'pose': 320, 'combined': 960} (separate per-session limits; total process VRAM is still measured independently)
- Date: 2026-09-29 20:14:04
- Each pipeline ran in its own process; every tick is CUDA-synchronized before the clock stops.
- Optimized column = REAL-TIME run (readers paced to 30 FPS, like cameras).
- Decode: optimized readers decode LIVE (software decode INCLUDED; this host: ~528 FPS of 1080p per thread, 12 logical CPUs).

| Metric | Naive Baseline (`FP32`, Sync `B=1`) | Optimized Pipeline (`FP16`, Async Batched) | Speedup / Delta |
| :--- | :---: | :---: | :---: |
| **Aggregate System Throughput (Total FPS)** | 10.8 FPS | 119.0 FPS | 11.01x |
| **Effective Per-Stream FPS (4 Streams)** | 2.7 FPS | 29.8 FPS | 11.01x |
| **Batch Loop Latency — Mean (ms)** | 370.0 | 27.7 | -92.5% |
| **Batch Loop Latency — P50 / P95 / P99 (ms)** | 363.3 / 441.1 / 510.2 | 27.2 / 31.1 / 43.7 | -91.4% |
| **Peak GPU VRAM Allocated / Reserved (MB)** | 751 (device-wide; ~219 net of 532 used by other apps) / 126 | 954 (device-wide; ~422 net of 532 used by other apps) / 65 | +27.1% |
| **Host CPU Utilization (%) (this process, share of all cores)** | 7.7% | 14.8% | +7.1 pts |
| **Dropped / Stale Frames Under Overload** | N/A (lags infinitely) | 30 dropped (99.2% of 120.0 offered FPS served); frame age P50/P99/max = 39.8/65.1/83.1 ms | Verified |

## Saturation capacity (readers unpaced: how fast can the pipeline go?)

| Metric | Naive Baseline | Optimized (saturated) | Speedup |
| :--- | :---: | :---: | :---: |
| **Total FPS** | 10.8 | 62.5 | 5.78x |
| **P99 batch-loop latency (ms)** | 510.2 | 127.7 | -75.0% |
| **Average batch size** | 1 | 4.00 | - |

## Where the optimized tick spends its time (real-time run, mean ms)

| pre_a | det | decode | sched | pre_b | pose |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2.53 | 18.70 | 3.90 | 0.09 | 0.02 | 0.55 |

Average batch size 3.39 | detections/tick 22.5 | Model B ROIs processed 63 over 1055 ticks | pose outputs decoded 57 | ROIs deferred by scheduler 0 of 63 candidates

## Targets

- Aggregate throughput >= 120 FPS (real-time): FAIL (119.0 FPS)
- P99 <= 33.3 ms (real-time): FAIL (43.7 ms)
- Peak VRAM <= 1536 MB: PASS (954 MB, device-wide, conservative: includes other apps via nvml; ~422 MB attributable to this pipeline)
- Sustained per-stream FPS >= 30 (real-time): FAIL (29.8)
- Pose outputs decoded: PASS (57 valid pose results; verify keypoint accuracy separately)