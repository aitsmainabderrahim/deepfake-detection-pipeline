"""
scripts/extract_faces.py

Batch face extraction from FaceForensics++ (or similar) video datasets.
Uses YOLOv8 face detector (Stage 1) to extract and save face crops.

Expected dataset layout:
    data_dir/
      original_sequences/actors/c23/videos/     ← REAL
      manipulated_sequences/
        Deepfakes/c23/videos/                   ← FAKE
        Face2Face/c23/videos/
        FaceSwap/c23/videos/
        NeuralTextures/c23/videos/

Output layout:
    output_dir/
      real/  *.png
      fake/  *.png

Usage:
    python scripts/extract_faces.py \\
        --data_dir /path/to/FaceForensics \\
        --output_dir data/faces \\
        --compression c23 \\
        --num_frames 30
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.data.face_extractor import YOLOFaceExtractor
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# FaceForensics++ path templates
FF_REAL = [
    "original_sequences/actors/{c}/videos",
    "original_sequences/youtube/{c}/videos",
]
FF_FAKE = [
    "manipulated_sequences/Deepfakes/{c}/videos",
    "manipulated_sequences/Face2Face/{c}/videos",
    "manipulated_sequences/FaceSwap/{c}/videos",
    "manipulated_sequences/NeuralTextures/{c}/videos",
    "manipulated_sequences/FaceShifter/{c}/videos",
]


def collect_videos(data_dir: Path, compression: str):
    videos = []
    for tmpl in FF_REAL:
        d = data_dir / tmpl.format(c=compression)
        if d.exists():
            for v in d.glob("*.mp4"):
                videos.append((v, 0))
    for tmpl in FF_FAKE:
        d = data_dir / tmpl.format(c=compression)
        if d.exists():
            for v in d.glob("*.mp4"):
                videos.append((v, 1))
    return videos


def process_one(args):
    video_path, label, output_dir, num_frames, detector_weights, conf = args
    try:
        extractor = YOLOFaceExtractor(
            weights=detector_weights,
            confidence=conf,
            target_size=224,
        )
        sub = output_dir / ("real" if label == 0 else "fake")
        sub.mkdir(parents=True, exist_ok=True)
        return extractor.process_video(str(video_path), str(sub),
                                       label=label, num_frames=num_frames)
    except Exception as e:
        logger.error(f"Error: {video_path}: {e}")
        return 0


def main():
    p = argparse.ArgumentParser(description="Extract faces from video dataset")
    p.add_argument("--data_dir",         required=True)
    p.add_argument("--output_dir",       default="data/faces")
    p.add_argument("--config",           default="configs/default.yaml")
    p.add_argument("--compression",      default="c23",
                   choices=["c0", "c23", "c40"])
    p.add_argument("--num_frames",       type=int, default=30)
    p.add_argument("--num_workers",      type=int, default=4)
    p.add_argument("--max_videos",       type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    det_weights = cfg.face_detector.weights
    det_conf    = cfg.face_detector.confidence

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    (output_dir / "real").mkdir(parents=True, exist_ok=True)
    (output_dir / "fake").mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning {data_dir} for {args.compression} videos ...")
    videos = collect_videos(data_dir, args.compression)

    if args.max_videos:
        videos = videos[:args.max_videos]

    real_n = sum(1 for _, l in videos if l == 0)
    fake_n = sum(1 for _, l in videos if l == 1)
    logger.info(f"Found {len(videos)} videos: {real_n} real, {fake_n} fake")

    tasks = [
        (v, l, output_dir, args.num_frames, det_weights, det_conf)
        for v, l in videos
    ]

    total_saved = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(process_one, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Extracting faces") as pbar:
            for fut in as_completed(futures):
                total_saved += fut.result()
                pbar.update(1)

    real_imgs = len(list((output_dir / "real").glob("*.png")))
    fake_imgs = len(list((output_dir / "fake").glob("*.png")))
    logger.info(
        f"\nExtraction complete!\n"
        f"  Real faces: {real_imgs}\n"
        f"  Fake faces: {fake_imgs}\n"
        f"  Output:     {output_dir}"
    )


if __name__ == "__main__":
    main()
