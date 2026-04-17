"""
scripts/download_weights.py

Download pretrained weights needed by the pipeline:
  1. yolov8n.pt          — General YOLOv8 (backbone for classifier)
  2. yolov8n-face.pt     — YOLOv8 fine-tuned on WiderFace (Stage 1 detector)

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --skip_face   # only download backbone
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path("weights")

# YOLOv8 official weights (auto-downloaded by ultralytics)
YOLO_VARIANTS = ["yolov8n", "yolov8s", "yolov8m"]

# YOLOv8-face: community model fine-tuned on WiderFace
# Source: https://github.com/akanametov/yolov8-face
FACE_MODEL_URL = (
    "https://github.com/akanametov/yolov8-face/releases/"
    "download/v0.0.0/yolov8n-face.pt"
)


def download_yolo_backbone(variant: str = "yolov8n"):
    """Download YOLOv8 backbone weights via ultralytics auto-download."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / f"{variant}.pt"
    if dest.exists():
        print(f"  ✓ Already exists: {dest}")
        return

    try:
        from ultralytics import YOLO
        print(f"  Downloading {variant}.pt via ultralytics ...")
        model = YOLO(f"{variant}.pt")   # triggers download to ~/.ultralytics
        # Copy from cache to weights/
        import shutil, glob
        patterns = [
            Path.home() / f".ultralytics/assets/{variant}.pt",
            Path(f"{variant}.pt"),
        ]
        for src in patterns:
            if Path(src).exists():
                shutil.copy(src, dest)
                print(f"  ✓ Saved: {dest}")
                return
        print(f"  ✓ {variant}.pt downloaded (check ~/.ultralytics/assets/)")
    except ImportError:
        print("  ultralytics not installed. Run: pip install ultralytics")


def download_face_model():
    """Download YOLOv8-face weights."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / "yolov8n-face.pt"

    if dest.exists():
        print(f"  ✓ Already exists: {dest}")
        return

    print(f"  Downloading yolov8n-face.pt from GitHub ...")
    try:
        def progress(block, block_size, total):
            done = block * block_size
            if total > 0:
                pct = min(100, done * 100 // total)
                print(f"\r  [{pct:3d}%] {done//1024}KB / {total//1024}KB  ", end="")

        urllib.request.urlretrieve(FACE_MODEL_URL, dest, reporthook=progress)
        print(f"\n  ✓ Saved: {dest}")
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        print(f"    Manual download: {FACE_MODEL_URL}")
        print(f"    Save to: {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--variant",    default="yolov8n",
                        choices=YOLO_VARIANTS,
                        help="YOLOv8 variant for backbone")
    parser.add_argument("--skip_face",  action="store_true",
                        help="Skip face detector download")
    parser.add_argument("--skip_backbone", action="store_true",
                        help="Skip backbone download")
    args = parser.parse_args()

    print("\n=== DeepFake YOLOv8 — Weight Downloader ===\n")

    if not args.skip_backbone:
        print(f"[1/2] YOLOv8 backbone ({args.variant}.pt):")
        download_yolo_backbone(args.variant)

    if not args.skip_face:
        print("\n[2/2] YOLOv8-face detector (yolov8n-face.pt):")
        download_face_model()

    print(f"\nDone. Weights in: {WEIGHTS_DIR.resolve()}/")
    print("\nNext steps:")
    print("  python scripts/extract_faces.py --data_dir /path/to/FaceForensics")
    print("  python scripts/train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
