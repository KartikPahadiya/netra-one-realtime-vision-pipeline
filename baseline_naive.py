"""baseline_naive.py — the unoptimized FP32 / synchronous / B=1 reference pipeline.

Intentionally keeps the production anti-patterns from the brief (sequential
decode blocking, unbatched B=1 inference, per-crop secondary calls, CPU-side
preprocessing inside ultralytics, no VRAM guardrails). It exists ONLY so
benchmark.py can measure the optimized pipeline against it.
"""
import time

import cv2
import torch
from ultralytics import YOLO


def run_naive_pipeline(video_paths, num_steps=300, secondary_every_box=5,
                       duration_s=None, warmup_steps=0, after_warmup=None):
    """Models are loaded ONCE. `warmup_steps` iterations run first and are
    discarded; `after_warmup()` is then called (benchmark starts its telemetry
    there) before the measured loop."""
    primary_detector = YOLO("yolo11n.pt").to("cuda")
    secondary_model = YOLO("yolo11n-pose.pt").to("cuda")
    caps = [cv2.VideoCapture(p) for p in video_paths]

    def step():
        n = 0
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    continue
            n += 1
            det_results = primary_detector(frame, verbose=False)      # B=1, FP32
            boxes = det_results[0].boxes.xyxy.cpu().numpy()
            for box in boxes[:secondary_every_box]:                   # unbatched loop
                x1, y1, x2, y2 = map(int, box)
                crop = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
                if crop.size > 0:
                    _ = secondary_model(crop, verbose=False)
        torch.cuda.synchronize()
        return n

    for _ in range(warmup_steps):
        step()
    if after_warmup:
        after_warmup()

    latencies_ms, frames = [], 0
    start = time.perf_counter()
    for _ in range(num_steps):
        if duration_s is not None and time.perf_counter() - start >= duration_s:
            break
        t0 = time.perf_counter()
        frames += step()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - start
    for cap in caps:
        cap.release()
    return {"latencies_ms": latencies_ms, "frames_processed": frames,
            "dropped_frames": 0, "wall_s": wall}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    s = run_naive_pipeline(args.videos, args.steps, warmup_steps=30)
    print(f"Naive baseline: {len(s['latencies_ms'])} steps, "
          f"mean {sum(s['latencies_ms'])/len(s['latencies_ms']):.1f} ms, frames {s['frames_processed']}")
