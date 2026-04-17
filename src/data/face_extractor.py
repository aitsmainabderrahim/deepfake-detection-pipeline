"""
src/data/face_extractor.py

Stage 1 — YOLOv8 Face Detector.

Uses a YOLOv8 model fine-tuned on WiderFace to detect faces in frames.
Falls back to OpenCV Haar cascade if YOLOv8 face weights are unavailable.

Pipeline per frame:
  1. Run YOLOv8 inference  → list of face bounding boxes
  2. Pad each box by scale_factor (capture forehead/chin context)
  3. Crop + resize to target_size × target_size
  4. Return list of (face_crop_bgr, original_box)
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# YOLOv8-face pretrained weights URL
YOLO_FACE_URL = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"


class YOLOFaceExtractor:
    """
    Face detector using YOLOv8 (fine-tuned on WiderFace).

    Args:
        weights:      Path to yolov8n-face.pt or similar.
        confidence:   Detection confidence threshold.
        target_size:  Output crop size (square).
        scale_factor: Padding multiplier around detected box.
        min_face_size: Minimum face dimension to keep (pixels).
        device:       'cpu' | 'cuda'.

    Usage:
        extractor = YOLOFaceExtractor('weights/yolov8n-face.pt')
        faces = extractor.extract_all(frame)
        # → List[(crop_bgr, (x1,y1,x2,y2))]
    """

    def __init__(
        self,
        weights:       str   = "weights/yolov8n-face.pt",
        confidence:    float = 0.4,
        target_size:   int   = 224,
        scale_factor:  float = 1.3,
        min_face_size: int   = 60,
        device:        str   = "cpu",
    ):
        self.target_size   = target_size
        self.scale_factor  = scale_factor
        self.min_face_size = min_face_size
        self.confidence    = confidence

        self._yolo = None
        self._haar = None
        self._load_detector(weights, confidence, device)

    # ── Public API ───────────────────────────────────────────────────────────

    def extract_all(
        self, frame: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int,int,int,int]]]:
        """
        Detect all faces and return cropped, resized images.

        Returns:
            List of (face_crop_bgr uint8 224×224, (x1,y1,x2,y2)) tuples.
            Sorted largest face first.
        """
        boxes = self._detect(frame)
        results = []
        for x1, y1, x2, y2 in boxes:
            w = x2 - x1
            h = y2 - y1
            if w < self.min_face_size or h < self.min_face_size:
                continue
            crop = self._pad_and_crop(frame, x1, y1, x2, y2)
            crop = cv2.resize(crop, (self.target_size, self.target_size),
                              interpolation=cv2.INTER_LANCZOS4)
            results.append((crop, (x1, y1, x2, y2)))

        # Sort by face area descending
        results.sort(key=lambda r: (r[1][2]-r[1][0]) * (r[1][3]-r[1][1]), reverse=True)
        return results

    def extract_largest(
        self, frame: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Tuple[int,int,int,int]]]:
        """Extract only the largest detected face. Returns None if no face."""
        faces = self.extract_all(frame)
        return faces[0] if faces else None

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        label: int,
        num_frames: int = 30,
    ) -> int:
        """
        Extract faces from a video file and save to disk.

        Args:
            video_path:  Input video (.mp4, .avi, ...).
            output_dir:  Directory to save face PNGs.
            label:       0=real, 1=fake (encoded in filename).
            num_frames:  Max frames to sample.

        Returns:
            Number of face images saved.
        """
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        cap   = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs  = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)

        saved = 0
        stem  = Path(video_path).stem

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            result = self.extract_largest(frame)
            if result is not None:
                crop, _ = result
                fname = out / f"{stem}_f{idx:05d}_l{label}.png"
                cv2.imwrite(str(fname), crop)
                saved += 1

        cap.release()
        return saved

    # ── Internal ─────────────────────────────────────────────────────────────

    def _load_detector(self, weights: str, confidence: float, device: str):
        if Path(weights).exists():
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(weights)
                self._yolo_conf = confidence
                self._yolo_device = device
                logger.info(f"[FaceExtractor] YOLOv8 loaded: {weights}")
                return
            except Exception as e:
                logger.warning(f"[FaceExtractor] YOLOv8 load failed: {e}")

        # Fallback: Haar cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        logger.warning("[FaceExtractor] Using Haar cascade fallback.")

    def _detect(self, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        if self._yolo is not None:
            return self._yolo_detect(frame)
        return self._haar_detect(frame)

    def _yolo_detect(self, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        results = self._yolo.predict(
            source=frame,
            conf=self._yolo_conf,
            verbose=False,
            device=self._yolo_device,
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _haar_detect(self, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0:
            return []
        return [(x, y, x+w, y+h) for x, y, w, h in faces]

    def _pad_and_crop(
        self, frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        H, W = frame.shape[:2]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        hw = int((x2 - x1) * self.scale_factor / 2)
        hh = int((y2 - y1) * self.scale_factor / 2)
        nx1 = max(0, cx - hw)
        ny1 = max(0, cy - hh)
        nx2 = min(W, cx + hw)
        ny2 = min(H, cy + hh)
        return frame[ny1:ny2, nx1:nx2].copy()
