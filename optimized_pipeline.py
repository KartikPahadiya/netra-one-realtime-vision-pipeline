"""optimized_pipeline.py — Netra-One optimized multi-stream engine.

    StreamReader x N --> LatestFrameQueue (depth 1, drop-oldest)
                               |
                       micro-batch collector (<= 5 ms window, starts at the
                       FIRST arriving frame; dead streams are not waited for)
                               |
                         GPU worker (main thread)
        1. CPU: resize ONLY, straight into a reused pinned uint8 buffer
        2. GPU: BGR->RGB, NCHW, float, /255 (one fused pass, no CPU float math)
        3. Model A via ORT io_binding on pre-allocated buffers (batch buckets)
        4. batched GPU decode + NMS (one D2H copy of final boxes only)
        5. ROI scheduler -> Model B on <= 4 person ROIs/tick at 256x256; decode keypoints

Design rules: nothing blocking in the GPU thread; all shapes are bucketed and
pre-warmed; all GPU buffers are allocated once at startup (VRAM discipline).
"""
import logging
import os
import queue
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger("netra.pipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE = 640                 # square fallback size
DET_HW = (384, 640)            # Model A input (H, W): rect letterbox for 16:9, multiples of 32
ROI_SIZE = 256                 # Model B input (crops are small; 640 is wasteful)
BATCH_TIMEOUT_MS = 2.0         # micro-batch window (starts at first arrival)
IDLE_WAIT_MS = 5.0             # maximum idle wait when no stream has a frame
VRAM_CAP_MB = 1536
CONF_THRESHOLD = 0.45
MIN_BOX_AREA = 40 * 40
MAX_ROIS_PER_TICK = 4
MODEL_B_COOLDOWN = 3
KEEP_CLASSES = (0, 2, 7)       # person, car, truck
ALIVE_WINDOW_S = 0.5           # a stream silent this long is not waited for
FPS = 30
USE_CUDA_GRAPH = os.environ.get("NETRA_CUDA_GRAPH") == "1"
SLOW_MS = float(os.environ.get("NETRA_SLOW_MS", 33.3))
def make_buckets(n: int):
    """Batch-size buckets 1,2,4,8... up to >= n. Batches are padded UP to the
    nearest bucket so only a handful of shapes ever exist (all pre-warmed)."""
    b, out = 1, []
    while True:
        out.append(b)
        if b >= n:
            return tuple(out)
        b *= 2


def bucket_for(n: int, buckets) -> int:
    for b in buckets:
        if b >= n:
            return b
    return buckets[-1]


# ---------------------------------------------------------------------------
# 1. Bounded latest-frame queue (drop-oldest backpressure)
# ---------------------------------------------------------------------------
class LatestFrameQueue:
    """Depth-1 queue: `put` overwrites the unread frame (latest-frame-wins)."""

    def __init__(self, maxsize: int = 1):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            try:
                self._q.put_nowait(item)
            except queue.Full:
                try:
                    self._q.get_nowait()          # drop the stale frame
                    self.dropped += 1
                except queue.Empty:
                    pass
                self._q.put_nowait(item)

    def get(self, timeout: float = 0.0):
        try:
            return self._q.get(timeout=timeout if timeout > 0 else 0)
        except queue.Empty:
            return None

    def size(self):
        return self._q.qsize()


_CLIPS = {}
_CLIPS_LOCK = threading.Lock()


def load_clip(path: str, max_frames: int = 600):
    """Decode a video ONCE into RAM (shared, read-only). Used to take software
    decode out of the measurement when the host has too few cores to decode
    4x1080p30 live (see benchmark.py --live-decode)."""
    with _CLIPS_LOCK:
        if path not in _CLIPS:
            cap, frames = cv2.VideoCapture(path), []
            while len(frames) < max_frames:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            cap.release()
            if not frames:
                raise IOError(f"cannot decode {path}")
            _CLIPS[path] = frames
        return _CLIPS[path]


# ---------------------------------------------------------------------------
# 2. Fault-resilient, real-time-paced stream reader
# ---------------------------------------------------------------------------
class StreamReader(threading.Thread):
    """One thread per stream.

    - Paced to `fps` (like a real camera) unless realtime=False (saturation
      mode: decode as fast as possible to measure pipeline capacity).
    - `synthetic://N` sources generate rolling-noise frames.
    - Initial open failure retries with backoff (camera down at boot).
    - File EOF: instant rewind. RTSP dropout: reconnect with backoff.
    - Corrupt/empty frames are skipped. Fault hooks (`blackout_until`,
      `corrupt_until`) let the benchmark simulate outages.
    Queue items are (frame_id, frame, capture_timestamp).
    """

    RETRY_S = 0.5

    def __init__(self, source, out_queue, stop_event, width=1920, height=1080,
                 fps=FPS, realtime=True, preload=False, index=0, n_streams=1, low_prio=False):
        super().__init__(daemon=True, name=f"reader-{source}")
        self.source, self.out, self.stop = source, out_queue, stop_event
        self.preload, self.index, self.n_streams = preload, index, n_streams
        self.low_prio = low_prio
        self.wrap_times, self.slow_reads = [], []      # diagnostics for stall analysis
        self._clip, self._pos = None, 0
        self.width, self.height, self.fps, self.realtime = width, height, fps, realtime
        self.frames_read = 0
        self.reconnects = 0
        self.corrupt_skipped = 0
        self.blackout_until = 0.0
        self.corrupt_until = 0.0
        self._in_blackout = False
        self._cap = None
        self._noise = None
        self._deadline = None
        self.det_hw = None

    def _open(self):
        if self.source.startswith("synthetic://"):
            return None
        if self.preload:
            self._clip = load_clip(self.source)
            self._pos = self.index * len(self._clip) // max(1, self.n_streams)  # desync streams
            return None
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            raise IOError(f"cannot open {self.source}")
        return cap

    def _synthetic_frame(self):
        if self._noise is None:
            self._noise = np.random.randint(0, 255, (self.height, self.width, 3),
                                            dtype=np.uint8)
        self._noise = np.roll(self._noise, 3, axis=1)   # new array each call
        return self._noise

    def _pace(self):
        """Absolute-deadline pacing: no drift, decode time is absorbed."""
        if not self.realtime and not self.source.startswith("synthetic://"):
            return
        now = time.perf_counter()
        if self._deadline is None:
            self._deadline = now
        self._deadline += 1.0 / self.fps
        wait = self._deadline - now
        if wait > 0:
            self.stop.wait(wait)
        elif wait < -0.25:                 # fell far behind: resync, don't burst
            self._deadline = now

    def _next_frame(self):
        now = time.perf_counter()
        if now < self.blackout_until:                       # simulated dropout
            if not self._in_blackout:
                self._in_blackout = True
                self.reconnects += 1
                logger.warning("[%s] simulated dropout", self.source)
            self.stop.wait(0.05)
            return None
        self._in_blackout = False

        if self.source.startswith("synthetic://"):
            return self.frames_read, self._synthetic_frame(), time.perf_counter()

        if self._clip is not None:
            ret, frame = True, self._clip[self._pos % len(self._clip)]
            self._pos += 1
            if self._pos % len(self._clip) == 0:
                self.wrap_times.append(time.perf_counter())
        else:
            ret, frame = self._cap.read()
        if ret and frame is not None and frame.ndim == 3:
            if now < self.corrupt_until and (self.frames_read % 2 == 0):
                self.corrupt_skipped += 1                   # simulated corruption
                self.frames_read += 1
                return None
            return self.frames_read, frame, time.perf_counter()

        if self.source.startswith("rtsp"):
            self.reconnects += 1
            logger.warning("[%s] dropout - reconnecting (%d)", self.source, self.reconnects)
            self.stop.wait(self.RETRY_S)
            self._cap.release()
            self._cap = cv2.VideoCapture(self.source)
            return None
        # local file: EOF / corrupt -> instant rewind
        self.wrap_times.append(time.perf_counter())
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return None

    def run(self):
        try:   # decoders are best-effort; the GPU-feeding thread must win the CPU
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        except Exception:
            if self.low_prio and os.name == "nt":
                try:
                    import ctypes
                    k32 = ctypes.windll.kernel32
                    k32.SetThreadPriority(k32.GetCurrentThread(), -1)   # BELOW_NORMAL
                except Exception:
                    pass
        while not self.stop.is_set():
            try:
                self._cap = self._open()
                break
            except Exception as e:
                self.reconnects += 1
                logger.warning("[%s] open failed (%s) - retry in %.1fs",
                               self.source, e, self.RETRY_S)
                self.stop.wait(self.RETRY_S)
        if self.stop.is_set():
            return
        while not self.stop.is_set():
            t_read = time.perf_counter()
            item = self._next_frame()
            dur = time.perf_counter() - t_read
            if dur > 0.25 and len(self.slow_reads) < 50:
                self.slow_reads.append((t_read, dur))
            if item is None:
                continue
            self.frames_read += 1
            if self.det_hw is not None:
                buf = np.empty((*self.det_hw, 3), np.uint8)
                meta = fill_letterbox_u8([item[1]], buf[None], self.det_hw)[0]
                item = item + (buf, meta)
            self.out.put(item)
            self._pace()
        if self._cap is not None:
            self._cap.release()


# ---------------------------------------------------------------------------
# 3. Letterbox geometry + resize-only preprocessing into uint8 buffers
# ---------------------------------------------------------------------------
def as_hw(size):
    """int -> (size, size); (H, W) stays as is."""
    return tuple(size) if isinstance(size, (tuple, list)) else (size, size)


def letterbox_meta(w: int, h: int, size=IMG_SIZE):
    """Scale + padding for a letterbox into `size` (int or (H, W))."""
    H, W = as_hw(size)
    r = min(W / w, H / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    return r, (W - nw) // 2, (H - nh) // 2


def to_original_coords(box, meta):
    r, pad_l, pad_t = meta
    return ((box[0] - pad_l) / r, (box[1] - pad_t) / r,
            (box[2] - pad_l) / r, (box[3] - pad_t) / r)


def fill_letterbox_u8(frames, stage: np.ndarray, size: int):
    """Resize each BGR frame into stage[i] ([size,size,3] uint8, gray-padded).
    This is ALL the CPU does: no float math, no channel shuffles.
    Returns per-frame letterbox metas."""
    metas = []
    for i, f in enumerate(frames):
        h, w = f.shape[:2]
        r, pl, pt = letterbox_meta(w, h, size)
        nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
        stage[i].fill(114)
        cv2.resize(f, (nw, nh), dst=stage[i, pt:pt + nh, pl:pl + nw],
                   interpolation=cv2.INTER_LINEAR)
        metas.append((r, pl, pt))
    return metas


# ---------------------------------------------------------------------------
# 4. VRAM guardrails + ORT sessions
# ---------------------------------------------------------------------------
def enforce_vram_cap(cap_mb: int = VRAM_CAP_MB):
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap_mb * 1e6 / total))
            torch.cuda.init()
    except Exception as e:
        logger.warning("torch VRAM fraction unavailable: %s", e)


def _trt_usable() -> bool:
    try:
        import ctypes
        import pathlib
        import onnxruntime as ort
        capi = pathlib.Path(ort.__file__).parent / "capi"
        for name in ("onnxruntime_providers_tensorrt.dll",
                     "libonnxruntime_providers_tensorrt.so",
                     "libonnxruntime_providers_tensorrt.dylib"):
            dll = capi / name
            if dll.exists():
                ctypes.CDLL(str(dll))
                return True
        return False
    except Exception:
        return False


def build_providers(cap_mb: int = VRAM_CAP_MB, use_trt: bool = False, arena_mb: int = None):
    """ORT providers built only from providers that really load. `arena_mb` is
    the hard per-session CUDA arena ceiling (gpu_mem_limit)."""
    import onnxruntime as ort
    available = ort.get_available_providers()
    providers = []
    if use_trt:
        trt_name = next((p for p in available if p.lower().startswith("tensorrt")), None)
        if trt_name and _trt_usable():
            providers.append((trt_name, {
                "device_id": 0,
                "trt_max_workspace_size": 256 * 1024 * 1024,
                "trt_engine_cache_enable": True,
                "trt_force_sequential_engine_build": True,
                "trt_fp16_enable": True,
            }))
        elif trt_name:
            logger.warning("TensorRT listed but its runtime is missing - CUDA EP only.")
    if "CUDAExecutionProvider" in available:
        arena = (arena_mb or cap_mb // 4) * 1024 * 1024
        providers.append(("CUDAExecutionProvider", {
            "gpu_mem_limit": arena,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
            **({"enable_cuda_graph": "1"} if USE_CUDA_GRAPH else {}),
        }))
    providers.append("CPUExecutionProvider")
    return providers


def _prep_ort_cuda():
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception as e:
        logger.warning("ort.preload_dlls() failed: %s", e)


def load_sessions(det_onnx: str, sec_onnx: str, providers):
    """`providers` is a list (shared) or a (det_list, sec_list) tuple."""
    import onnxruntime as ort
    _prep_ort_cuda()
    prov_det, prov_sec = providers if isinstance(providers, tuple) else (providers, providers)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # CPU-side threads: keep them off the decode threads' cores, and never
    # busy-spin (Colab has only 2 vCPUs; spinning starves the GPU-feeding thread)
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    det = ort.InferenceSession(det_onnx, opts, providers=prov_det)
    sec = ort.InferenceSession(sec_onnx, opts, providers=prov_sec)
    return det, sec


# ---------------------------------------------------------------------------
# 5. Model runner
# ---------------------------------------------------------------------------
class ModelRunner:
    """Wraps one ORT session; discovers I/O dtypes and the output shape for
    each input size by real dummy runs (doubles as provider warmup)."""

    def __init__(self, session, probe_sizes=(DET_HW,)):
        self.s = session
        inp, out = session.get_inputs()[0], session.get_outputs()[0]
        self.in_name, self.out_name = inp.name, out.name
        self.in_np_dtype = np.float16 if "float16" in inp.type else np.float32
        self.out_np_dtype = np.float16 if "float16" in out.type else np.float32
        try:
            import torch
            self.in_torch_dtype = torch.float16 if self.in_np_dtype == np.float16 else torch.float32
            self.out_torch_dtype = torch.float16 if self.out_np_dtype == np.float16 else torch.float32
        except ImportError:
            self.in_torch_dtype = self.out_torch_dtype = None
        self.has_cuda = "CUDAExecutionProvider" in session.get_providers()
        self.out_shapes = {}          # (H, W) -> (channels, anchors)
        for size in probe_sizes:
            hw = as_hw(size)
            try:
                dummy = np.zeros((1, 3, hw[0], hw[1]), dtype=self.in_np_dtype)
                o = self.s.run([self.out_name], {self.in_name: dummy})[0]
                self.out_shapes[hw] = tuple(o.shape[1:])
            except Exception as e:
                logger.warning("input size %s unsupported by model: %s", hw, e)

    def supports(self, size):
        return as_hw(size) in self.out_shapes

    def run(self, tensor_np):
        return self.s.run([self.out_name], {self.in_name: tensor_np})[0]


# ---------------------------------------------------------------------------
# 6. YOLO output decoding
# ---------------------------------------------------------------------------
def decode_detections(output: np.ndarray, conf_thr: float = 0.25):
    """CPU decoder. output [B, 84, A] -> list of [N,6] (x1,y1,x2,y2,conf,cls)."""
    preds = output.astype(np.float32).transpose(0, 2, 1)
    results = []
    for p in preds:
        boxes, scores = p[:, :4], p[:, 4:]
        cls, conf = scores.argmax(1), scores.max(1)
        mask = (conf >= conf_thr) & np.isin(cls, KEEP_CLASSES)
        if not mask.any():
            results.append(np.zeros((0, 6), np.float32))
            continue
        cx, cy, w, h = boxes[mask].T
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        keep = _nms(xyxy, conf[mask], iou_thr=0.45)
        results.append(np.concatenate(
            [xyxy[keep], conf[mask][keep, None], cls[mask][keep, None]], 1).astype(np.float32))
    return results



def decode_pose_batch(output, n_valid: int, metas, crop_shapes, conf_thr: float = 0.25):
    """Decode one highest-confidence YOLO pose candidate per ROI.

    Expected YOLO11-pose output is [B, 56, A] = xywh + person score + 17*(x,y,score).
    Only the selected 56-value candidate per ROI is copied from GPU to CPU.
    Keypoints are returned in each original crop's coordinate system.
    Unsupported output layouts fail closed with None entries, not fake poses.
    """
    if n_valid <= 0:
        return []
    try:
        import torch
        if isinstance(output, torch.Tensor):
            if output.ndim != 3:
                return [None] * n_valid
            # ORT export layout: [B, channels, anchors]. Select the best anchor on GPU.
            if output.shape[1] < 56:
                return [None] * n_valid
            pred = output[:n_valid].float().permute(0, 2, 1)
            scores = pred[..., 4]
            best_idx = scores.argmax(dim=1)
            rows = torch.arange(n_valid, device=output.device)
            best = pred[rows, best_idx].detach().cpu().numpy()
        else:
            arr = np.asarray(output)
            if arr.ndim != 3 or arr.shape[1] < 56:
                return [None] * n_valid
            pred = arr[:n_valid].astype(np.float32).transpose(0, 2, 1)
            best_idx = pred[:, :, 4].argmax(axis=1)
            best = pred[np.arange(n_valid), best_idx]
    except Exception:
        arr = np.asarray(output)
        if arr.ndim != 3 or arr.shape[1] < 56:
            return [None] * n_valid
        pred = arr[:n_valid].astype(np.float32).transpose(0, 2, 1)
        best_idx = pred[:, :, 4].argmax(axis=1)
        best = pred[np.arange(n_valid), best_idx]

    results = []
    for i in range(n_valid):
        row = best[i].astype(np.float32, copy=False)
        if row.size < 56 or float(row[4]) < conf_thr:
            results.append(None)
            continue
        r, pad_l, pad_t = metas[i]
        h, w = crop_shapes[i]
        kp = row[5:56].reshape(17, 3).copy()
        kp[:, 0] = np.clip((kp[:, 0] - pad_l) / max(r, 1e-9), 0, max(0, w - 1))
        kp[:, 1] = np.clip((kp[:, 1] - pad_t) / max(r, 1e-9), 0, max(0, h - 1))
        results.append({
            "pose_confidence": float(row[4]),
            "keypoints": [{"x": float(x), "y": float(y), "confidence": float(c)}
                          for x, y, c in kp],
        })
    return results


def _nms(xyxy, scores, iou_thr, top=300):
    order = scores.argsort()[::-1][:top]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0]); yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2]); yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        area_r = (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


def decode_detections_gpu(out_t, n_valid: int, conf_thr: float = 0.25, max_pre_nms: int = 500):
    """Batched GPU decode: one class/conf pass, one batched NMS, ONE D2H copy.
    out_t: [Bbucket, 84, A] CUDA tensor; only the first n_valid rows are real."""
    try:
        import torch
        from torchvision.ops import batched_nms
    except Exception:
        return decode_detections(out_t[:n_valid].float().cpu().numpy(), conf_thr)
    p = out_t[:n_valid].float().permute(0, 2, 1)             # [B, A, 84]
    conf, cls = p[..., 4:].max(dim=-1)                        # [B, A]
    keep_cls = torch.zeros(p.shape[-1] - 4, dtype=torch.bool, device=p.device)
    keep_cls[list(KEEP_CLASSES)] = True
    mask = (conf >= conf_thr) & keep_cls[cls]
    b_idx, a_idx = mask.nonzero(as_tuple=True)
    empty = np.zeros((0, 6), np.float32)
    if b_idx.numel() == 0:
        return [empty] * n_valid
    # Apply the pre-NMS limit PER IMAGE. A single global top-k lets a crowded
    # stream consume every candidate slot and starve detections from other streams.
    per_image_keep = []
    for image_id in range(n_valid):
        idx = torch.nonzero(b_idx == image_id, as_tuple=False).flatten()
        if idx.numel() > max_pre_nms:
            image_scores = conf[b_idx[idx], a_idx[idx]]
            idx = idx[image_scores.topk(max_pre_nms).indices]
        if idx.numel():
            per_image_keep.append(idx)
    if not per_image_keep:
        return [empty] * n_valid
    selected = torch.cat(per_image_keep)
    b_idx, a_idx = b_idx[selected], a_idx[selected]

    cx, cy, w, h = p[b_idx, a_idx, :4].unbind(-1)
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    c, k = conf[b_idx, a_idx], cls[b_idx, a_idx]
    # NMS must be independent per image AND class. Using only b_idx suppresses
    # different classes; using only class IDs would suppress boxes across streams.
    num_classes = p.shape[-1] - 4
    nms_groups = b_idx * num_classes + k
    keep = batched_nms(xyxy, c, nms_groups, 0.45)
    packed = torch.cat([xyxy[keep], c[keep, None], k[keep, None].float(),
                        b_idx[keep, None].float()], -1).cpu().numpy()
    img = packed[:, 6].astype(int)
    return [packed[img == i, :6] if (img == i).any() else empty for i in range(n_valid)]


# ---------------------------------------------------------------------------
# 7. Model B ROI scheduler
# ---------------------------------------------------------------------------
class ROIScheduler:
    """Which ROIs get Model B this tick.

    - Gating: conf >= 0.45 and area >= min_area.
    - Per-stream cooldown: a stream is eligible at most every K ticks, and the
      cooldown starts ONLY if that stream actually got served (no starvation).
    - Fairness: ROIs are picked round-robin across streams (largest first
      within a stream), so one busy camera cannot monopolise the batch.
    - Hard cap `max_rois` per tick: GPU work per tick is bounded no matter how
      many people appear -> P99 and VRAM are bounded (crowd-surge immunity).
      Excess candidates are deferred (counted), never queued.
    """

    def __init__(self, cooldown=MODEL_B_COOLDOWN, max_rois=MAX_ROIS_PER_TICK,
                 conf=CONF_THRESHOLD, min_area=MIN_BOX_AREA):
        self.cooldown, self.max_rois = cooldown, max_rois
        self.conf, self.min_area = conf, min_area
        self._tick = 0
        self._last_run = {}
        self.candidates = 0
        self.selected = 0

    @staticmethod
    def _area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    def select(self, detections):
        self._tick += 1
        per_stream = {}
        for sid, dets in detections.items():
            if self._tick - self._last_run.get(sid, -10**9) < self.cooldown or len(dets) == 0:
                continue
            # Model B is a pose model: vehicle detections are useful to Model A,
            # but sending cars/trucks into the pose model wastes ROI budget.
            good = dets[(dets[:, 4] >= self.conf) & (dets[:, 5].astype(np.int32) == 0)]
            areas = (good[:, 2] - good[:, 0]) * (good[:, 3] - good[:, 1])
            good, areas = good[areas >= self.min_area], areas[areas >= self.min_area]
            if len(good):
                per_stream[sid] = good[np.argsort(-areas)]
        order = sorted(per_stream, key=lambda s: -self._area(per_stream[s][0]))
        chosen, idx, served = [], {s: 0 for s in per_stream}, set()
        progressed = True
        while progressed and len(chosen) < self.max_rois:
            progressed = False
            for s in order:
                if len(chosen) >= self.max_rois:
                    break
                if idx[s] < len(per_stream[s]):
                    chosen.append((s, per_stream[s][idx[s]]))
                    idx[s] += 1
                    served.add(s)
                    progressed = True
        for s in served:
            self._last_run[s] = self._tick
        self.candidates += sum(len(v) for v in per_stream.values())
        self.selected += len(chosen)
        chosen.sort(key=lambda sb: -self._area(sb[1]))
        return chosen

    @property
    def deferred(self):
        return self.candidates - self.selected


# ---------------------------------------------------------------------------
# 8. The pipeline
# ---------------------------------------------------------------------------
_WIN_TIMER = False


def _win_timer(on: bool):
    """Windows' default timer tick is ~15.6 ms, which makes Event.wait()/sleep()
    round up and adds 10+ ms of jitter to reader pacing and the collector.
    timeBeginPeriod(1) restores 1 ms resolution; also raise our priority a notch."""
    global _WIN_TIMER
    if os.name != "nt":
        return
    try:
        import ctypes
        if on and not _WIN_TIMER:
            ctypes.windll.winmm.timeBeginPeriod(1)
            _WIN_TIMER = True
            try:
                import psutil
                psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            except Exception:
                pass
        elif not on and _WIN_TIMER:
            ctypes.windll.winmm.timeEndPeriod(1)
            _WIN_TIMER = False
    except Exception:
        pass


def _torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


STAGES = ("pre_a", "det", "decode", "sched", "pre_b", "pose")


class OptimizedPipeline:
    def __init__(self, sources, det_onnx, sec_onnx, vram_cap_mb=VRAM_CAP_MB,
                 use_trt=False, realtime=True, max_rois=MAX_ROIS_PER_TICK,
                 surge_per_stream=0, preload=False, det_shape="rect",
                 win_timer=(os.name == "nt"), ffmpeg_threads=0, reader_prio=False, fixed_batch=True):
        """Opt-in switches (all OFF = the configuration that measured well on the RTX 2050):
        det_shape="rect"   Model A at 384x640 instead of padding to 640x640
        win_timer=True     Windows 1 ms timer resolution + above-normal priority
        ffmpeg_threads=N   limit FFmpeg decoder threads per capture (0 = library default)"""
        self.win_timer = win_timer
        # fixed_batch=True: each model only ever sees ONE input shape (Model A: batch=N streams,
        # Model B: batch=max_rois; short batches are padded). ORT's CUDA provider pays a
        # multi-second cuDNN re-planning cost on every input-shape change (measured: 2.6 s for
        # Model A, 1.4-2.1 s for Model B), so shape changes at runtime are forbidden.
        self.fixed_batch = fixed_batch
        if ffmpeg_threads:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"threads;{int(ffmpeg_threads)}"
        self.sources = sources
        n = len(sources)
        self.stop = threading.Event()
        self.queues = [LatestFrameQueue(1) for _ in sources]
        self.readers = [StreamReader(s, q, self.stop, realtime=realtime, preload=preload,
                                     index=i, n_streams=n, low_prio=reader_prio)
                        for i, (s, q) in enumerate(zip(sources, self.queues))]
        self.scheduler = ROIScheduler(max_rois=max_rois)
        self.surge_per_stream = surge_per_stream
        self.det_buckets = (n,) if fixed_batch else make_buckets(n)
        self.sec_buckets = (max_rois,) if fixed_batch else make_buckets(max_rois)

        enforce_vram_cap(vram_cap_mb)
        # Budget two separate ORT CUDA arenas below the requested ceiling, leaving
        # headroom for CUDA context, workspaces, and fixed staging/output buffers.
        # PyTorch's memory fraction does not govern ORT allocations; these provider
        # arena limits are the corresponding guardrail for ONNX Runtime.
        det_arena_mb = max(64, min(640, int(vram_cap_mb * 0.45)))
        sec_arena_mb = max(64, min(320, int(vram_cap_mb * 0.25)))
        max_arena_total = max(128, int(vram_cap_mb * 0.75))
        if det_arena_mb + sec_arena_mb > max_arena_total:
            sec_arena_mb = max(64, max_arena_total - det_arena_mb)
        self.ort_arena_budget_mb = {"detector": det_arena_mb, "pose": sec_arena_mb,
                                    "combined": det_arena_mb + sec_arena_mb}
        prov = (build_providers(vram_cap_mb, use_trt, arena_mb=det_arena_mb),
                build_providers(vram_cap_mb, use_trt, arena_mb=sec_arena_mb))
        self.det, self.sec = load_sessions(det_onnx, sec_onnx, prov)
        logger.info("det providers: %s", self.det.get_providers())
        want = DET_HW if det_shape == "rect" else (IMG_SIZE, IMG_SIZE)
        self.det_runner = ModelRunner(self.det, (want, (IMG_SIZE, IMG_SIZE)))
        self.det_hw = want if self.det_runner.supports(want) else (IMG_SIZE, IMG_SIZE)
        for r in self.readers:
            r.det_hw = self.det_hw
        self.sec_runner = ModelRunner(self.sec, (ROI_SIZE, IMG_SIZE))
        self.roi_size = ROI_SIZE if self.sec_runner.supports(ROI_SIZE) else IMG_SIZE
        self.use_gpu = (self.det_runner.has_cuda and self.sec_runner.has_cuda
                        and _torch_cuda_available())
        self._init_buffers()

        self.last_seen = [time.perf_counter()] * n
        self.last_frame_ts = [0.0] * n
        self.reset_stats()

    # -- buffers ------------------------------------------------------------
    def _init_buffers(self):
        na, nb = self.det_buckets[-1], self.sec_buckets[-1]
        dh, dw = self.det_hw
        R = self.roi_size
        if not self.use_gpu:
            self._np_a = np.zeros((na, dh, dw, 3), np.uint8)
            self._np_b = np.zeros((nb, R, R, 3), np.uint8)
            logger.info("CPU fallback path active")
            return
        import torch
        self._pin_a = torch.empty((na, dh, dw, 3), dtype=torch.uint8, pin_memory=True)
        self._pin_b = torch.empty((nb, R, R, 3), dtype=torch.uint8, pin_memory=True)
        self._np_a, self._np_b = self._pin_a.numpy(), self._pin_b.numpy()
        self._gpu_a = torch.empty((na, dh, dw, 3), dtype=torch.uint8, device="cuda")
        self._gpu_b = torch.empty((nb, R, R, 3), dtype=torch.uint8, device="cuda")
        self._gbuf = {}
        for tag, runner, size, buckets in (("a", self.det_runner, self.det_hw, self.det_buckets),
                                           ("b", self.sec_runner, R, self.sec_buckets)):
            hw = as_hw(size)
            C, A = runner.out_shapes[hw]
            for b in buckets:
                x = torch.zeros((b, 3, hw[0], hw[1]), dtype=runner.in_torch_dtype, device="cuda")
                out = torch.empty((b, C, A), dtype=runner.out_torch_dtype, device="cuda")
                bind = runner.s.io_binding()
                bind.bind_input(runner.in_name, "cuda", 0, runner.in_np_dtype,
                                list(x.shape), x.data_ptr())
                bind.bind_output(runner.out_name, "cuda", 0, runner.out_np_dtype,
                                 list(out.shape), out.data_ptr())
                self._gbuf[(tag, b)] = (x, out, bind)
        logger.info("GPU path: pinned uint8 staging, GPU normalize, io_binding on "
                    "pre-allocated buffers, Model A %dx%d (rect), det buckets %s, "
                    "sec buckets %s @%dpx", dh, dw, self.det_buckets, self.sec_buckets, R)

    def reset_stats(self):
        self.slow = []
        self.latencies_ms, self.frame_ages_ms, self.batch_sizes = [], [], []
        self.stage_ms = {k: [] for k in STAGES}
        self.frames_processed = self.rois_processed = self.ticks = 0
        self.detections_total = 0
        self.pose_results_total = 0
        self.latest_pose_results = {}
        self.per_stream_frames = [0] * len(self.sources)
        self._drop0 = sum(q.dropped for q in self.queues)
        self.stalls, self._det_split, self._t0 = [], None, time.perf_counter()
        self._prev_bucket = getattr(self, "_prev_bucket", {})
        self._changed = {"a": False, "b": False}
        self.scheduler.candidates = self.scheduler.selected = 0

    # -- micro-batch collector ---------------------------------------------
    def collect_batch(self):
        n = len(self.queues)
        batch = {}
        t0 = time.perf_counter()
        idle_deadline = t0 + IDLE_WAIT_MS / 1000.0
        window_deadline = None
        while True:
            now = time.perf_counter()
            for sid in range(n):
                if sid in batch:
                    continue
                item = self.queues[sid].get(0)
                if item is not None:
                    batch[sid] = item
                    self.last_seen[sid] = now
                    if window_deadline is None:
                        window_deadline = now + BATCH_TIMEOUT_MS / 1000.0
            active = [s for s in range(n) if now - self.last_seen[s] < ALIVE_WINDOW_S]
            if all(s in batch for s in active) and batch:
                break                                   # every live stream is in
            if window_deadline is not None:
                if now >= window_deadline:
                    break
            elif now >= idle_deadline:
                break
            time.sleep(0.0004)
        return batch

    # -- helpers ------------------------------------------------------------
    def _sync(self):
        if self.use_gpu:
            import torch
            torch.cuda.synchronize()

    def _infer(self, tag, runner, frames, size, buckets, pre=None):
        """-> (output, t_pre_ms, t_run_ms). GPU: resize -> pinned u8 -> H2D ->
        GPU normalize -> ORT io_binding. CPU: same math in numpy."""
        b = len(frames)
        t0 = time.perf_counter()
        if not self.use_gpu:
            stage = self._np_a if tag == "a" else self._np_b
            metas = fill_letterbox_u8(frames, stage, size)
            x = np.ascontiguousarray(stage[:b, :, :, ::-1].transpose(0, 3, 1, 2)
                                     ).astype(runner.in_np_dtype) / np.asarray(255, runner.in_np_dtype)
            t1 = time.perf_counter()
            out = runner.run(x.astype(runner.in_np_dtype))
            return out, metas, (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3
        import torch
        bucket = bucket_for(b, buckets)
        prev = self._prev_bucket.get(tag)
        self._changed[tag] = prev is not None and prev != bucket
        self._prev_bucket[tag] = bucket
        stage_np = self._np_a if tag == "a" else self._np_b
        pin = self._pin_a if tag == "a" else self._pin_b
        gpu = self._gpu_a if tag == "a" else self._gpu_b
        if pre is not None:
            for i, (buf, _) in enumerate(pre):
                np.copyto(stage_np[i], buf)
            metas = [m for _, m in pre]
        else:
            metas = fill_letterbox_u8(frames, stage_np, size)
        gpu[:bucket].copy_(pin[:bucket], non_blocking=True)
        x, out, bind = self._gbuf[(tag, bucket)]
        x.copy_(gpu[:bucket].flip(-1).permute(0, 3, 1, 2))     # BGR->RGB, NCHW, dtype
        x.mul_(1.0 / 255.0)
        torch.cuda.current_stream().synchronize()              # ORT has its own stream
        t1 = time.perf_counter()
        runner.s.run_with_iobinding(bind)
        t2 = time.perf_counter()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        self._split = ((t2 - t1) * 1e3, (t3 - t2) * 1e3)     # (ORT call, CUDA sync)
        return out, metas, (t1 - t0) * 1e3, (t3 - t1) * 1e3

    def warmup_shapes(self):
        """Run every (model, bucket) once: kernels are compiled before timing."""
        try:
            if self.use_gpu:
                import torch
                for _ in range(3):
                    for (tag, b), (x, out, bind) in self._gbuf.items():
                        (self.det_runner if tag == "a" else self.sec_runner).s.run_with_iobinding(bind)
                self._sync()
            else:
                for b in self.det_buckets:
                    self.det_runner.run(np.zeros((b, 3, self.det_hw[0], self.det_hw[1]), self.det_runner.in_np_dtype))
                for b in self.sec_buckets:
                    self.sec_runner.run(np.zeros((b, 3, self.roi_size, self.roi_size),
                                                 self.sec_runner.in_np_dtype))
            logger.info("shape warmup done")
        except Exception as e:
            logger.warning("shape warmup skipped: %s", e)

    def self_check(self, frame):
        """Run ONE real frame through the fast path (pinned u8 -> GPU normalize ->
        io_binding) and through a plain reference path (numpy preprocessing +
        ordinary session.run). Prints whether they agree."""
        try:
            out_g, _, _, _ = self._infer("a", self.det_runner, [frame], self.det_hw, self.det_buckets)
            dets_g = (decode_detections_gpu(out_g, 1) if self.use_gpu
                      else decode_detections(out_g))[0]
            raw_g = (out_g[:1].float().cpu().numpy() if self.use_gpu
                     else out_g[:1].astype(np.float32))
            tmp = np.zeros((1, self.det_hw[0], self.det_hw[1], 3), np.uint8)
            fill_letterbox_u8([frame], tmp, self.det_hw)
            x = (tmp[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
            ref = self.det_runner.run(np.ascontiguousarray(x.astype(self.det_runner.in_np_dtype)))
            dets_r = decode_detections(ref)[0]
            d_score = float(np.abs(raw_g[:, 4:] - ref.astype(np.float32)[:, 4:]).max())
            ok = d_score < 0.05 and abs(len(dets_g) - len(dets_r)) <= 1
            print(f"[selfcheck] fast-path dets={len(dets_g)} reference dets={len(dets_r)} "
                  f"max score diff={d_score:.4f} -> {'OK' if ok else 'MISMATCH (fast path is wrong!)'}",
                  flush=True)
            return ok
        except Exception as e:
            print(f"[selfcheck] failed: {e}", flush=True)
            return False

    def _inject_surge(self, det_map, n):
        """Crowd-surge test: n fake high-confidence persons per stream."""
        rng = np.random.default_rng(1)
        for sid in det_map:
            x = rng.uniform(20, self.det_hw[1] - 100, n); y = rng.uniform(20, self.det_hw[0] - 170, n)
            fake = np.stack([x, y, x + 60, y + 150, np.full(n, .9), np.zeros(n)], 1).astype(np.float32)
            det_map[sid] = np.concatenate([det_map[sid], fake], 0)

    # -- one tick -----------------------------------------------------------
    def run_tick(self):
        # Measure the full tick, including waiting for the micro-batch collector.
        # Starting the clock after collect_batch() hides queue/collection latency
        # from the P95/P99 metric and makes the benchmark look faster than reality.
        t_start = time.perf_counter()
        batch = self.collect_batch()
        if not batch:
            return False
        self._changed = {"a": False, "b": False}
        sids = sorted(batch)
        frames = [batch[s][1] for s in sids]
        stamps = [batch[s][2] for s in sids]
        pre = ([(batch[s][3], batch[s][4]) for s in sids]
               if all(len(batch[s]) > 4 for s in sids) else None)

        out_a, metas, pre_a, run_a = self._infer("a", self.det_runner, frames, self.det_hw, self.det_buckets, pre=pre)
        self._det_split = getattr(self, "_split", None)
        t = time.perf_counter()
        dets = (decode_detections_gpu(out_a, len(frames)) if self.use_gpu
                else decode_detections(out_a))
        decode_ms = (time.perf_counter() - t) * 1e3
        det_map = {s: d for s, d in zip(sids, dets)}
        if self.surge_per_stream:
            self._inject_surge(det_map, self.surge_per_stream)
        self.detections_total += sum(len(d) for d in det_map.values())

        t = time.perf_counter()
        picks = self.scheduler.select(det_map)
        crops = []
        crop_infos = []
        for sid, box in picks:
            frame = batch[sid][1]
            x1, y1, x2, y2 = to_original_coords(box, metas[sids.index(sid)])
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            if x2 - x1 > 1 and y2 - y1 > 1:
                crops.append(frame[y1:y2, x1:x2])
                crop_infos.append({"sid": sid, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        sched_ms = (time.perf_counter() - t) * 1e3

        pre_b = run_b = 0.0
        self.latest_pose_results = {}  # bounded: keep only the most recent tick's poses
        if crops:
            pose_out, pose_metas, pre_b, run_b = self._infer(
                "b", self.sec_runner, crops, self.roi_size, self.sec_buckets)
            t_pose_decode = time.perf_counter()
            pose_items = decode_pose_batch(
                pose_out, len(crops), pose_metas,
                [crop.shape[:2] for crop in crops], conf_thr=0.25)
            # Account for output decoding in the secondary-model stage.
            run_b += (time.perf_counter() - t_pose_decode) * 1e3
            for info, pose in zip(crop_infos, pose_items):
                if pose is None:
                    continue
                for kp in pose["keypoints"]:
                    kp["x"] += info["x1"]
                    kp["y"] += info["y1"]
                pose["stream_id"] = info["sid"]
                pose["roi_box_xyxy"] = [info["x1"], info["y1"], info["x2"], info["y2"]]
                self.latest_pose_results.setdefault(info["sid"], []).append(pose)
                self.pose_results_total += 1
            self.rois_processed += len(crops)

        end = time.perf_counter()
        lat = (end - t_start) * 1e3
        if lat > SLOW_MS and len(self.slow) < 500:
            self.slow.append((lat, pre_a, run_a, decode_ms, sched_ms, pre_b, run_b, len(frames), len(crops)))
        if lat > 250.0 and len(self.stalls) < 50:
            self.stalls.append({"end": end, "lat_ms": lat,
                                "stage": dict(zip(STAGES, (pre_a, run_a, decode_ms, sched_ms, pre_b, run_b))),
                                "det_split": self._det_split, "shape_changed": dict(self._changed),
                                "frames": len(frames), "rois": len(crops)})
        self.latencies_ms.append(lat)
        self.frame_ages_ms.extend((end - ts) * 1e3 for ts in stamps)
        self.batch_sizes.append(len(frames))
        for k, v in zip(STAGES, (pre_a, run_a, decode_ms, sched_ms, pre_b, run_b)):
            self.stage_ms[k].append(v)
        for s in sids:
            self.per_stream_frames[s] += 1
            self.last_frame_ts[s] = end
        self.frames_processed += len(frames)
        self.ticks += 1
        return True

    def dropped_frames(self):
        return sum(q.dropped for q in self.queues) - self._drop0

    def inject_faults(self, schedule):
        """schedule: [(t_offset_s, 'blackout'|'corrupt', stream_id, duration_s)]"""
        def worker():
            t0 = time.perf_counter()
            for off, kind, sid, dur in sorted(schedule):
                if self.stop.wait(max(0.0, off - (time.perf_counter() - t0))):
                    return
                until = time.perf_counter() + dur
                if kind == "blackout":
                    self.readers[sid].blackout_until = until
                else:
                    self.readers[sid].corrupt_until = until
        threading.Thread(target=worker, daemon=True).start()

    def run(self, num_steps=300, duration_s=None, warmup_ticks=0, after_warmup=None):
        if self.win_timer:
            _win_timer(True)
        self.warmup_shapes()
        for r in self.readers:
            r.start()
        w0 = time.perf_counter()
        n = 0
        while n < warmup_ticks and time.perf_counter() - w0 < 90:
            if self.run_tick():
                n += 1
        if warmup_ticks:
            self.reset_stats()
        if after_warmup:
            after_warmup()
        start = time.perf_counter()
        n = 0
        while n < num_steps:
            if duration_s is not None and time.perf_counter() - start >= duration_s:
                break
            if self.run_tick():
                n += 1
        wall = time.perf_counter() - start
        return self.shutdown(wall)

    def shutdown(self, wall_s=None):
        if self.win_timer:
            _win_timer(False)
        self.stop.set()
        for r in self.readers:
            if r.is_alive():
                r.join(timeout=2.0)
        now = time.perf_counter()
        wraps = sorted(t for r in self.readers for t in r.wrap_times)
        stall_info = []
        for st in self.stalls[:10]:
            near = min((abs(st["end"] - w) for w in wraps), default=None)
            stall_info.append({"t_s": round(st["end"] - self._t0, 1), "lat_ms": round(st["lat_ms"]),
                               "stage_ms": {k: round(v) for k, v in st["stage"].items() if v > 50},
                               "frames": st["frames"], "rois": st["rois"],
                               "shape_changed": st["shape_changed"],
                               "det_ort_call_ms": round(st["det_split"][0]) if st["det_split"] else None,
                               "det_cuda_sync_ms": round(st["det_split"][1]) if st["det_split"] else None,
                               "nearest_reader_rewind_s": round(near, 1) if near is not None else None})
        slow_reads = [(round(t - self._t0, 1), round(d, 2)) for r in self.readers for t, d in r.slow_reads]
        age = np.asarray(self.frame_ages_ms) if self.frame_ages_ms else np.zeros(1)
        return {
            "latencies_ms": self.latencies_ms,
            "frames_processed": self.frames_processed,
            "rois_processed": self.rois_processed,
            "pose_results_total": self.pose_results_total,
            "latest_pose_results": self.latest_pose_results,
            "latest_pose_count": sum(len(v) for v in self.latest_pose_results.values()),
            "ticks": self.ticks,
            "dropped_frames": self.dropped_frames(),
            "reconnects": sum(r.reconnects for r in self.readers),
            "wall_s": wall_s,
            "avg_batch": float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0,
            "frame_age_p50_ms": float(np.percentile(age, 50)),
            "frame_age_p99_ms": float(np.percentile(age, 99)),
            "frame_age_max_ms": float(age.max()),
            "stage_ms_mean": {k: (float(np.mean(v)) if v else 0.0) for k, v in self.stage_ms.items()},
            "per_stream_frames": self.per_stream_frames,
            "stream_silent_s": [round(now - t, 2) if t else None for t in self.last_frame_ts],
            "detections_per_tick": self.detections_total / max(1, self.ticks),
            "rois_candidates": self.scheduler.candidates,
            "rois_deferred": self.scheduler.deferred,
            "in_dtype": np.dtype(self.det_runner.in_np_dtype).name,
            "roi_size": self.roi_size,
            "max_rois": self.scheduler.max_rois,
            "det_shape": list(self.det_hw), "fixed_batch": self.fixed_batch,
            "slow_ticks": self.slow,
            "gpu_path": self.use_gpu,
            "det_providers": list(self.det.get_providers()),
            "sec_providers": list(self.sec.get_providers()),
            "ort_arena_budget_mb": self.ort_arena_budget_mb,
            "stall_count": len(self.stalls), "stalls": stall_info,
            "reader_slow_reads": sorted(slow_reads)[:10], "reader_rewinds": len(wraps),
        }


if __name__ == "__main__":
    # Quick A/B runner (~20 s): find which optimization helps or hurts on YOUR machine.
    #   python optimized_pipeline.py --video assets/sample_1080p.mp4
    #   python optimized_pipeline.py --video assets/sample_1080p.mp4 --det-shape rect
    #   python optimized_pipeline.py --video assets/sample_1080p.mp4 --win-timer
    #   python optimized_pipeline.py --video assets/sample_1080p.mp4 --ffmpeg-threads 1
    import argparse

    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="1080p clip looped on 4 readers (default: synthetic)")
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--det", default="yolo11n.onnx")
    parser.add_argument("--sec", default="yolo11n-pose.onnx")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--unpaced", action="store_true", help="saturation mode")
    parser.add_argument("--det-shape", choices=["square", "rect"], default="rect")
    parser.add_argument("--win-timer", action="store_true", default=(os.name == "nt"))
    parser.add_argument("--ffmpeg-threads", type=int, default=0)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--reader-prio", action="store_true", help="Windows: readers at below-normal priority")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="ablation: pad to buckets 1/2/4(/8) instead of one fixed shape (expect multi-second stalls)")
    args = parser.parse_args()
    srcs = ([args.video] * args.streams if args.video
            else [f"synthetic://{i}" for i in range(args.streams)])
    pipe = OptimizedPipeline(srcs, args.det, args.sec, realtime=not args.unpaced,
                             preload=args.preload, det_shape=args.det_shape,
                             win_timer=args.win_timer, ffmpeg_threads=args.ffmpeg_threads,
                             reader_prio=args.reader_prio, fixed_batch=not args.dynamic_batch)
    WU = int(os.environ.get("NETRA_WARMUP", 30))
    st = pipe.run(num_steps=args.steps, warmup_ticks=WU)
    lat = np.asarray(st.pop("latencies_ms"))
    print(f"\nconfig: fixed_batch={not args.dynamic_batch} shape={args.det_shape} win_timer={args.win_timer} "
          f"ffmpeg_threads={args.ffmpeg_threads} preload={args.preload} gpu_path={st['gpu_path']}")
    print(f"ticks={len(lat)}  latency ms  mean={lat.mean():.1f}  P50={np.percentile(lat,50):.1f}  "
          f"P95={np.percentile(lat,95):.1f}  P99={np.percentile(lat,99):.1f}  max={lat.max():.1f}")
    print(f"fps={st['frames_processed']/st['wall_s']:.1f}  avg_batch={st['avg_batch']:.2f}  "
          f"dropped={st['dropped_frames']}  frame_age_p99={st['frame_age_p99_ms']:.1f} ms")
    print("stage ms:", {k: round(v, 2) for k, v in st["stage_ms_mean"].items()})
    if st["slow_ticks"]:
        a = np.array(st["slow_ticks"])
        print(f"slow ticks >{SLOW_MS} ms: {len(a)}/{len(lat)} | mean {a[:,0].mean():.1f} | "
              f"pre_a/det/decode/sched/pre_b/pose = {np.round(a[:,1:7].mean(0),1).tolist()} | "
              f"unaccounted {np.round((a[:,0]-a[:,1:7].sum(1)).mean(),1)} ms | "
              f"frames {a[:,7].mean():.1f} rois {a[:,8].mean():.1f}")
    print(f"stalls (>250 ms ticks): {st['stall_count']} | reader rewinds: {st['reader_rewinds']} | "
          f"reader reads >250 ms: {len(st['reader_slow_reads'])}")
    for x in st["stalls"]:
        print("  stall", x)
    if st["reader_slow_reads"]:
        print("  slow reader reads (t_s, seconds):", st["reader_slow_reads"])