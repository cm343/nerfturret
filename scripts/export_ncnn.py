#!/usr/bin/env python3
"""
export_ncnn.py — export the YOLOv8 weights to NCNN for fast on-Pi inference.

The released .pt weights are ~YOLOv8m; running them through PyTorch on a Pi is
slow. NCNN is the fastest CPU backend on ARM, and Ultralytics' YOLO() loads the
exported model transparently, so detection.py needs no change — just point
DEFAULT_WEIGHTS (or the "local" backend) at the produced .ncnn model.

Run this once on the Pi (or anywhere with ultralytics installed):

    python scripts/export_ncnn.py
    python scripts/export_ncnn.py --weights .../generalized_40_class/best.pt --imgsz 320

It writes a sibling "<name>_ncnn_model" directory next to the .pt; pass that
directory as the weights path to LocalYoloDetector.
"""

import argparse

from ultralytics import YOLO

# Import the project defaults so this script and the runtime agree.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from detection import DEFAULT_WEIGHTS, IMG_SIZE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 weights to NCNN.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="export input size")
    args = parser.parse_args()

    print(f"[export] loading {args.weights}")
    model = YOLO(args.weights)
    out = model.export(format="ncnn", imgsz=args.imgsz)
    print(f"[export] wrote NCNN model: {out}")
    print("[export] point LocalYoloDetector's weights at that directory.")


if __name__ == "__main__":
    main()
