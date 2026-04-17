"""
app/webcam_detector.py

Real-Time DeepFake Detector — YOLOv8 Two-Stage Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Per frame:
  1. Stage 1 → YOLOv8 detects face bounding boxes
  2. For each face:
       a. Crop + normalize to 224×224
       b. Compute FFT + texture feature maps
       c. Stage 2 → YOLOv8 backbone + DeepFakeHead → P(fake)
       d. Temporal smoothing (10-frame moving average)
  3. Annotate: draw box, REAL/FAKE label, confidence bar
  4. HUD: FPS, face count, model info

Controls:
  Q / ESC  — Quit
  S        — Save snapshot
  F        — Toggle FFT overlay
  +/-      — Raise / lower confidence threshold

Usage:
  python app/webcam_detector.py --checkpoint weights/checkpoints/best_model.pth
  python app/webcam_detector.py --checkpoint weights/checkpoints/best_model.pth --camera 1
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import collections
import logging
from pathlib import Path
from typing import Dict, Deque, Optional

import cv2
import numpy as np
import torch

from src.utils.config import load_config
from src.models.deepfake_model import build_model
from src.data.face_extractor import YOLOFaceExtractor
from src.features.frequency import compute_fft_feature_map
from src.features.texture import compute_texture_feature_map

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Color palette ─────────────────────────────────────────────────────────────
COLOR_REAL  = (50,  220, 50)
COLOR_FAKE  = (30,  30,  220)
COLOR_UNK   = (160, 160, 160)
COLOR_WHITE = (255, 255, 255)
FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO   = cv2.FONT_HERSHEY_SIMPLEX


# ── Inference Engine ──────────────────────────────────────────────────────────

class DeepFakeEngine:
    """Wraps Stage-2 model for fast single-face inference."""

    def __init__(self, checkpoint: str, cfg, device: torch.device):
        self.device = device
        self.cfg    = cfg

        self.model = build_model(cfg)
        ckpt = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(device).eval()

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)

        # Warm-up
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        with torch.no_grad():
            self.model(dummy)
        logger.info(f"[Engine] Model ready | device={device}")

    @torch.no_grad()
    def predict(self, face_bgr: np.ndarray) -> float:
        """Returns P(fake) ∈ [0,1] for a single BGR face crop."""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # RGB tensor
        t = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
        t = ((t - self.mean) / self.std).unsqueeze(0).to(self.device)

        # Feature tensor (FFT + texture)
        x_feat = None
        if self.cfg.features.use_fft or self.cfg.features.use_srm:
            fft = compute_fft_feature_map(rgb, size=224)
            tex = compute_texture_feature_map(rgb, size=224)
            combined = np.concatenate([fft, tex], axis=0)
            x_feat = torch.from_numpy(combined).unsqueeze(0).to(self.device)

        logit, _ = self.model(t, x_feat)
        return torch.sigmoid(logit).item()


# ── Temporal Smoother ─────────────────────────────────────────────────────────

class Smoother:
    def __init__(self, window: int = 10):
        self.buf: Deque[float] = collections.deque(maxlen=window)
    def update(self, v: float) -> float:
        self.buf.append(v)
        return float(np.mean(self.buf))
    def reset(self): self.buf.clear()


# ── Frame Drawing ─────────────────────────────────────────────────────────────

def draw_result(frame, box, prob, threshold=0.5):
    x1, y1, x2, y2 = map(int, box)
    is_fake = prob >= threshold
    color   = COLOR_FAKE if is_fake else COLOR_REAL
    label   = "FAKE" if is_fake else "REAL"
    conf    = prob if is_fake else 1 - prob

    # Corner-bracket bounding box
    L = max(12, min((x2-x1)//5, (y2-y1)//5, 28))
    t = 3
    for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px,py), (px+dx*L,py), color, t)
        cv2.line(frame, (px,py), (px,py+dy*L), color, t)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (*color, 50), 1)

    # Label badge
    text = f"{label}  {conf*100:.1f}%"
    (tw,th),_ = cv2.getTextSize(text, FONT, 0.75, 2)
    ly = max(0, y1-8)
    cv2.rectangle(frame, (x1, ly-th-10), (x1+tw+12, ly+4), color, -1)
    cv2.putText(frame, text, (x1+6, ly-3), FONT, 0.75, COLOR_WHITE, 2, cv2.LINE_AA)

    # Confidence bar (green=real, red=fake, threshold marker)
    bx, by, bw, bh = x1, y2+8, x2-x1, 10
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (40,40,40), -1)
    fw = int(prob * bw)
    if fw > 0:
        cv2.rectangle(frame, (bx,by), (bx+fw,by+bh), COLOR_FAKE, -1)
    if bw - fw > 0:
        cv2.rectangle(frame, (bx+fw,by), (bx+bw,by+bh), COLOR_REAL, -1)
    tx = bx + int(threshold * bw)
    cv2.line(frame, (tx,by-2), (tx,by+bh+2), COLOR_WHITE, 2)
    return frame


def draw_hud(frame, fps, n_faces, threshold):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0,0), (w,46), (12,12,12), -1)
    frame = cv2.addWeighted(ov, 0.75, frame, 0.25, 0)

    cv2.putText(frame, "DEEPFAKE DETECTOR  [YOLOv8]",
                (10, 30), FONT, 0.85, (0,200,255), 2, cv2.LINE_AA)

    info = f"FPS:{fps:.1f}  Faces:{n_faces}  Thr:{threshold:.2f}"
    (iw,_),_ = cv2.getTextSize(info, FONT_MONO, 0.55, 1)
    cv2.putText(frame, info, (w-iw-10, 28), FONT_MONO, 0.55,
                (200,200,200), 1, cv2.LINE_AA)

    # Key hints
    hints = "[Q] Quit  [S] Save  [F] FFT  [+/-] Threshold"
    cv2.putText(frame, hints, (10, h-10), FONT_MONO, 0.45,
                (130,130,130), 1, cv2.LINE_AA)
    return frame


def draw_fft_inset(frame, face_bgr):
    """Draw FFT spectrum of the largest face in bottom-right corner."""
    from src.features.frequency import compute_fft_spectrum
    rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    spec = compute_fft_spectrum(rgb)
    spec_u8 = (spec * 255).astype(np.uint8)
    colored = cv2.applyColorMap(spec_u8, cv2.COLORMAP_INFERNO)
    h, w  = frame.shape[:2]
    th, tw = h // 4, w // 4
    colored = cv2.resize(colored, (tw, th))
    frame[h-th-10:h-10, w-tw-10:w-10] = colored
    cv2.putText(frame, "FFT", (w-tw-10, h-th-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)
    return frame


# ── Main App ──────────────────────────────────────────────────────────────────

class WebcamApp:
    def __init__(self, checkpoint: str, cfg, camera: int):
        self.cfg       = cfg
        self.threshold = cfg.realtime.confidence_threshold
        self.show_fft  = False
        self.snap_n    = 0

        self.device = torch.device(
            cfg.project.device if torch.cuda.is_available() else "cpu"
        )

        # Stage 1 — YOLOv8 face detector
        self.face_extractor = YOLOFaceExtractor(
            weights=cfg.face_detector.weights,
            confidence=cfg.face_detector.confidence,
            target_size=cfg.data.image_size,
            scale_factor=cfg.face_detector.scale_factor,
            min_face_size=cfg.face_detector.min_face_size,
            device=cfg.face_detector.device,
        )

        # Stage 2 — DeepFake classifier
        self.engine = DeepFakeEngine(checkpoint, cfg, self.device)

        # Per-face temporal smoothers (indexed by slot)
        self.smoothers: Dict[int, Smoother] = {}

        self.camera = camera
        self._fps_times = collections.deque(maxlen=30)

    def run(self):
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.realtime.display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.realtime.display_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        logger.info("Webcam started. Press Q to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                self._fps_times.append(time.perf_counter())

                annotated = self._process(frame)
                cv2.imshow("DeepFake Detector — YOLOv8", annotated)

                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key, frame):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Session ended.")

    def _process(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()

        # Stage 1: detect faces
        faces = self.face_extractor.extract_all(frame)

        largest_face = None
        for i, (face_crop, box) in enumerate(faces):
            if i == 0:
                largest_face = face_crop

            # Ensure smoother for this slot
            if i not in self.smoothers:
                self.smoothers[i] = Smoother(
                    self.cfg.realtime.temporal_smoothing_window
                )

            # Stage 2: classify
            raw_prob    = self.engine.predict(face_crop)
            smooth_prob = self.smoothers[i].update(raw_prob)

            annotated = draw_result(annotated, box, smooth_prob, self.threshold)

        if not faces:
            self.smoothers.clear()
            h, w = annotated.shape[:2]
            msg  = "No face detected"
            (tw,_),_ = cv2.getTextSize(msg, FONT, 0.8, 2)
            cv2.putText(annotated, msg, ((w-tw)//2, h//2),
                        FONT, 0.8, COLOR_UNK, 2, cv2.LINE_AA)

        # FFT inset
        if self.show_fft and largest_face is not None:
            annotated = draw_fft_inset(annotated, largest_face)

        # HUD
        fps = self._get_fps()
        annotated = draw_hud(annotated, fps, len(faces), self.threshold)

        return annotated

    def _handle_key(self, key: int, frame) -> bool:
        if key in (ord("q"), 27):
            return True
        elif key == ord("s"):
            self._snap(frame)
        elif key == ord("f"):
            self.show_fft = not self.show_fft
            logger.info(f"FFT overlay: {'ON' if self.show_fft else 'OFF'}")
        elif key in (ord("+"), ord("=")):
            self.threshold = min(0.95, self.threshold + 0.05)
            logger.info(f"Threshold: {self.threshold:.2f}")
        elif key == ord("-"):
            self.threshold = max(0.05, self.threshold - 0.05)
            logger.info(f"Threshold: {self.threshold:.2f}")
        return False

    def _snap(self, frame):
        Path("results/snapshots").mkdir(parents=True, exist_ok=True)
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = f"results/snapshots/snap_{ts}_{self.snap_n:03d}.jpg"
        cv2.imwrite(path, frame)
        self.snap_n += 1
        logger.info(f"Saved: {path}")

    def _get_fps(self) -> float:
        t = self._fps_times
        if len(t) < 2: return 0.0
        return (len(t) - 1) / (t[-1] - t[0] + 1e-9)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepFake Detector — YOLOv8 Webcam")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to classification model checkpoint (.pth)")
    parser.add_argument("--config",    default="configs/default.yaml")
    parser.add_argument("--camera",    type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.threshold is not None:
        cfg.realtime.confidence_threshold = args.threshold

    app = WebcamApp(
        checkpoint=args.checkpoint,
        cfg=cfg,
        camera=args.camera,
    )
    app.run()


if __name__ == "__main__":
    main()
