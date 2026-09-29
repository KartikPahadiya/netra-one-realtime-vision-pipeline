"""Fast smoke tests for optimized_pipeline.py. No GPU or real model files required."""
import os
import sys
import time
import types
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimized_pipeline as op

# 1. Latest-frame queue: bounded and drop-oldest.
q = op.LatestFrameQueue(1)
q.put(("a", 1)); q.put(("b", 2)); q.put(("c", 3))
assert q.get() == ("c", 3) and q.dropped == 2
print("PASS 1: drop-oldest queue")

# 2. Letterbox geometry and resize into reusable uint8 buffer.
meta = op.letterbox_meta(1920, 1080)
assert abs(meta[0] - 1 / 3) < 1e-6 and meta[1] == 0 and meta[2] == 140, meta
assert op.to_original_coords(np.array([100., 200., 200., 300.]), meta) == (300.0, 180.0, 600.0, 480.0)
frame = np.full((1080, 1920, 3), 200, np.uint8)
stage = np.zeros((2, 640, 640, 3), np.uint8)
op.fill_letterbox_u8([frame], stage, 640)
assert stage[0, 0, 0, 0] == 114 and stage[0, 320, 320, 0] == 200
assert stage[0, 139, 5, 0] == 114 and stage[0, 140, 5, 0] == 200
print("PASS 2: letterbox geometry + resize-into-view")

# 3. Primary detector output decoding.
out = np.zeros((1, 84, 8400), np.float32)
out[0, :4, 0] = [320, 320, 100, 200]
out[0, 4, 0] = .99
out[0, 20, 0] = .5
# Keep the test prediction a person (class 0).
out[0, 4, 0] = .99
out[0, 5:, 0] = 0
out[0, 4, 0] = .99
out[0, 5, 0] = .01
out[0, 4, 0] = .99
# Decoder uses max over class channels; class 0 is the first score channel.
d = op.decode_detections(out, 0.25)[0]
assert len(d) == 1 and d[0, 5] == 0 and abs(d[0, 0] - 270) < 1e-3, d
print("PASS 3: decode_detections")

# 4. Pose output decoding: 17 keypoints map from letterboxed ROI to crop coordinates.
pose_out = np.zeros((1, 56, 10), np.float32)
pose_out[0, 4, 0] = .9
for k in range(17):
    pose_out[0, 5 + 3 * k, 0] = 128.0
    pose_out[0, 5 + 3 * k + 1, 0] = 128.0
    pose_out[0, 5 + 3 * k + 2, 0] = .8
pose_meta = op.letterbox_meta(100, 200, 256)
poses = op.decode_pose_batch(pose_out, 1, [pose_meta], [(200, 100)])
assert poses[0] is not None and len(poses[0]["keypoints"]) == 17
assert abs(poses[0]["keypoints"][0]["x"] - 50.0) < 1e-4
assert abs(poses[0]["keypoints"][0]["y"] - 100.0) < 1e-4
print("PASS 4: pose/keypoint decoding + inverse letterbox mapping")

# 5. Scheduler gating, cooldown, cap, fairness, and person-only pose routing.
big = np.array([[0, 0, 100, 100, .9, 0]], np.float32)
small = np.array([[0, 0, 20, 20, .99, 0]], np.float32)
vehicle = np.array([[0, 0, 200, 200, .99, 2]], np.float32)
s = op.ROIScheduler(cooldown=2, max_rois=3, conf=.45, min_area=1000)
assert [p[0] for p in s.select({0: np.vstack([big, vehicle]), 1: small})] == [0]
assert s.select({0: big}) == []
assert len(s.select({0: big})) == 1
many = np.array([[0, 0, 100 + i, 100 + i, .9, 0] for i in range(10)], np.float32)
s2 = op.ROIScheduler(cooldown=1, max_rois=3)
picks = s2.select({0: many, 1: big})
assert {p[0] for p in picks} == {0, 1} and len(picks) == 3, picks
s3 = op.ROIScheduler(cooldown=5, max_rois=8)
surge = {i: np.tile(big, (10, 1)) for i in range(4)}
assert len(s3.select(surge)) == 8 and s3.deferred == 32
starve = op.ROIScheduler(cooldown=100, max_rois=1)
starve.select({0: big, 1: big})
assert len(starve.select({0: big, 1: big})) == 1
print("PASS 5: scheduler (person gating, cooldown, fairness, cap, no starvation)")

# Create a small real clip for low-cost, repeatable reader/batching tests.
tmpdir = tempfile.mkdtemp()
clip_tmp = os.path.join(tmpdir, "t.mp4")
wr = cv2.VideoWriter(clip_tmp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
for i in range(20):
    wr.write(np.full((48, 64, 3), i * 10, np.uint8))
wr.release()
op.load_clip(clip_tmp)  # preload before starting reader threads so startup is deterministic

# Fake ONNX Runtime sessions for end-to-end pipeline tests without model files/GPU.
class FakeSession:
    def __init__(self, pose=False):
        self.pose = pose
    def get_providers(self):
        return ["CPUExecutionProvider"]
    def get_inputs(self):
        return [types.SimpleNamespace(name="images", type="tensor(float)", shape=["b", 3, "h", "w"])]
    def get_outputs(self):
        return [types.SimpleNamespace(name="output0", type="tensor(float)")]
    def run(self, names, feeds):
        x = feeds["images"]
        b, _, h, w = x.shape
        anchors = h // 8 * (w // 8) + h // 16 * (w // 16) + h // 32 * (w // 32)
        if self.pose:
            o = np.zeros((b, 56, anchors), np.float32)
            o[:, 4, 0] = .9
            for k in range(17):
                o[:, 5 + 3 * k, 0] = w / 2
                o[:, 5 + 3 * k + 1, 0] = h / 2
                o[:, 5 + 3 * k + 2, 0] = .9
        else:
            o = np.zeros((b, 84, anchors), np.float32)
            o[:, :4, 0] = [w / 2, h / 2, w / 3, h / 2]
            o[:, 4, 0] = .9  # person confidence, class 0
        return [o]

op.load_sessions = lambda d, s, p: (FakeSession(False), FakeSession(True))
op.build_providers = lambda *a, **k: ["CPUExecutionProvider"]
op.enforce_vram_cap = lambda *a, **k: None

# 6. End-to-end four synthetic streams + a dead stream; Model B returns keypoints.
srcs = [clip_tmp] * 4 + ["nonexistent_dead_stream.mp4"]
pipe = op.OptimizedPipeline(srcs, "fake-det.onnx", "fake-pose.onnx", preload=True, win_timer=False)
assert pipe.use_gpu is False
t0 = time.perf_counter()
st = pipe.run(num_steps=60)
assert st["ticks"] == 60 and st["frames_processed"] >= 60 and st["rois_processed"] > 0
assert st["pose_results_total"] > 0, st["pose_results_total"]
assert pipe.readers[-1].reconnects > 0, "dead stream not retried"
assert not any(r.is_alive() for r in pipe.readers)
assert st["dropped_frames"] == pipe.dropped_frames()
assert st["avg_batch"] >= 1.5, f"batching not observed: {st['avg_batch']}"
print(f"PASS 6: end-to-end 60 ticks, avg batch {st['avg_batch']:.2f}, "
      f"pose results {st['pose_results_total']}, dead-stream retries {pipe.readers[-1].reconnects} "
      f"({time.perf_counter()-t0:.1f}s)")

# 7. Empty collector exits within its bounded idle window.
p2 = op.OptimizedPipeline(["synthetic://9"], "fake-det.onnx", "fake-pose.onnx", win_timer=False)
t0 = time.perf_counter(); e = p2.collect_batch(); dt = (time.perf_counter() - t0) * 1e3
p2.shutdown(); assert e == {} and dt < 50, dt
print(f"PASS 7: collector idle timeout ({dt:.1f} ms; configured {op.IDLE_WAIT_MS:.1f} ms)")

# 8. A blackout on one stream does not stall the healthy stream; recovery resumes.
p3 = op.OptimizedPipeline([clip_tmp, clip_tmp], "fake-det.onnx", "fake-pose.onnx", preload=True, win_timer=False)
p3.inject_faults([(0.3, "blackout", 1, 1.0)])
st = p3.run(duration_s=2.0, num_steps=10**9)
f0, f1 = st["per_stream_frames"]
assert f0 > 35 and f1 < f0 * .8 and f1 > 5, (f0, f1)
print(f"PASS 8: blackout isolated: healthy stream {f0} frames, faulty {f1}, recovered")

# 9. Preloaded clips wrap around and each stream begins at a different offset.
p4 = op.OptimizedPipeline([clip_tmp, clip_tmp], "fake-det.onnx", "fake-pose.onnx", preload=True, win_timer=False)
st = p4.run(duration_s=1.5, num_steps=10**9)
assert st["frames_processed"] > 30 and st["ticks"] > 20, st["ticks"]
assert p4.readers[0]._pos != p4.readers[1]._pos or p4.readers[0].frames_read > 20
print(f"PASS 9: preloaded clip ({st['frames_processed']} frames, decode excluded)")

print("\nALL SMOKE TESTS PASSED")
