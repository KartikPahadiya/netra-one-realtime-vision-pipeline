"""export_models.py — export YOLO11 PyTorch weights to ONNX / TensorRT.

    python export_models.py                                    # FP16 ONNX (default, dynamic batch/H/W)
    python export_models.py --precision fp32                   # FP32 ONNX
    python export_models.py --format engine --precision fp16   # TensorRT FP16 engine
    python export_models.py --format engine --precision int8   # TensorRT INT8 (PTQ)

After every ONNX export the weights are inspected, so "FP16" in the benchmark
report is a verified fact, not a label. (Ultralytics FP16 ONNX keeps FP32
input/output tensors and FP16 weights/compute inside; that is expected.)
"""
import argparse
from collections import Counter


def verify_onnx_precision(path: str) -> str:
    try:
        import onnx
    except ImportError:
        return "unverified (pip install onnx)"
    m = onnx.load(path)
    c = Counter(onnx.TensorProto.DataType.Name(i.data_type) for i in m.graph.initializer)
    fp16, fp32 = c.get("FLOAT16", 0), c.get("FLOAT", 0)
    return f"FP16 weights: {fp16} tensors | FP32 weights: {fp32} tensors"


def export(weights: str, fmt: str, precision: str, imgsz: int = 640) -> str:
    from ultralytics import YOLO
    model = YOLO(weights)
    half = precision in ("fp16", "int8")
    if fmt == "onnx":
        if precision == "int8":
            raise SystemExit("INT8 needs a TensorRT engine: use --format engine --precision int8")
        # dynamic=True -> dynamic batch AND height/width (Model B runs at 256px)
        out = model.export(format="onnx", imgsz=imgsz, dynamic=True, half=half, simplify=True)
        info = verify_onnx_precision(out)
        print(f"[export] {weights} -> {out}   [{info}]")
        if precision == "fp16" and "FP16 weights: 0" in info:
            print("[export] WARNING: requested FP16 but the ONNX has no FP16 weights. "
                  "Re-run on a machine/runtime with a CUDA GPU (device=0).")
    else:
        out = model.export(format="engine", imgsz=imgsz, dynamic=True, half=half,
                           int8=(precision == "int8"), workspace=1.0, batch=4)
        print(f"[export] {weights} -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11n-pose.pt"])
    p.add_argument("--format", choices=["onnx", "engine"], default="onnx")
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()
    for w in a.models:
        export(w, a.format, a.precision, a.imgsz)
