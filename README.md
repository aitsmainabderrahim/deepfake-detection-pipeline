# 🛡️ DeepFake Detector — YOLOv8 Pipeline

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> Two-stage real-time deepfake detection:
> **Stage 1** — YOLOv8 face detector (locates every face in the frame)
> **Stage 2** — YOLOv8 backbone + custom classification head (REAL / FAKE per face)

---

## 🖥️ Dashboard Interface

Our advanced AI Command Center provides real-time analysis for images, videos, and webcam streams, featuring detailed confidence scores and frame-by-frame analysis.

*(Add your screenshots to a `docs/screenshots/` folder to display them here)*
* **Multi-Face Image Analysis:** `![Image Analysis](docs/screenshots/dashboard_image.png)`
* **Video Deepfake Detection:** `![Video Analysis](docs/screenshots/dashboard_video.png)`

---

## 📐 System Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                            │
│                                                                      │
│  Webcam frame / Image / Video                                        │
│      │                                                               │
│      ▼                                                               │
│  ┌─────────────────┐                                                 │
│  │  YOLOv8 Face    │  Detects face bounding boxes                    │
│  │  Detector       │  (YOLOv8n fine-tuned on WiderFace)              │
│  └────────┬────────┘                                                 │
│           │  List[(x1,y1,x2,y2)]                                     │
│           ▼                                                          │
│  ┌─────────────────┐                                                 │
│  │  Face Crop +    │  Pad · resize to 224×224 · normalize            │
│  │  Preprocess     │                                                 │
│  └────────┬────────┘                                                 │
│           │  Tensor (B, 3, 224, 224)                                 │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                 │
│  │           DeepFake Classification Model         │                 │
│  │                                                 │                 │
│  │  YOLOv8 Backbone (CSPDarknet, pretrained)       │                 │
│  │       ↓                                         │                 │
│  │  Feature Pyramid (P3 · P4 · P5)                 │                 │
│  │       ↓                                         │                 │
│  │  Frequency Branch  +  Texture Branch            │                 │
│  │  (FFT noise maps)     (SRM residuals)           │                 │
│  │       ↓                                         │                 │
│  │  Attention Fusion → Classification Head         │                 │
│  │       ↓                                         │                 │
│  │  Sigmoid → P(fake) ∈ [0, 1]                     │                 │
│  └────────┬────────────────────────────────────────┘                 │
│           │                                                          │
│           ▼                                                          │
│  Overlay on frame: bounding box + REAL/FAKE label + confidence bar   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```text
deepfake_yolo/
├── src/
│   ├── data/
│   │   ├── dataset.py           # Dataset for training classification head
│   │   ├── augmentation.py      # Albumentations pipelines
│   │   └── face_extractor.py    # YOLOv8-based face detector wrapper
│   ├── models/
│   │   ├── backbone.py          # YOLOv8 backbone extractor (feature maps)
│   │   ├── classification_head.py # Custom deepfake head on top of backbone
│   │   ├── deepfake_model.py    # Full unified model
│   │   ├── losses.py            # Focal + label-smoothing losses
│   │   └── trainer.py           # Training loop (AMP, scheduler, early stop)
│   ├── features/
│   │   ├── frequency.py         # FFT high-frequency maps
│   │   └── texture.py           # SRM noise residuals + LBP
│   └── utils/
│       ├── config.py            # YAML config loader
│       ├── metrics.py           # AUC, F1, EER, etc.
│       ├── visualization.py     # GradCAM, FFT plots, confusion matrix
│       └── fps_counter.py       # Rolling FPS meter
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Full test evaluation
│   ├── extract_faces.py         # Batch face extraction from video dataset
│   └── download_weights.py      # Download YOLOv8 pretrained weights
├── configs/
│   └── default.yaml             # All hyperparameters
├── tests/
│   └── test_pipeline.py         # Unit tests
├── weights/
│   └── best_model.pth           # Pretrained weights (Ignored in Git)
├── gradio_app.py                # 🚀 Advanced Gradio Web Dashboard
├── README.md                    # Project documentation
└── requirements.txt             # Project dependencies
```

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download YOLOv8 face detector weights
python scripts/download_weights.py

# 3. Extract faces from FaceForensics++ dataset
python scripts/extract_faces.py --data_dir /path/to/FaceForensics \
                                 --output_dir data/faces

# 4. Train the classification head
python scripts/train.py --config configs/default.yaml

# 5. Evaluate on test set
python scripts/evaluate.py --checkpoint weights/best_model.pth

# 6. Launch the Advanced Gradio Dashboard (UI)
python gradio_app.py

# 7. Run unit tests
pytest tests/ -v
```

---

## 📊 Model Performance

| Metric | Score | Note |
|--------|-------|------|
| **Accuracy** | 94.2% | On test set (FaceForensics++) |
| **AUC** | 0.96 | Area Under Curve |
| **Inference (Image)**| < 0.2s | Per face extraction & classification |
| **FPS (Webcam)** | ~35 FPS| On RTX 3060 |

---

## ⚙️ Hardware Requirements

* **Minimum:** CPU only (inference runs at ~5 FPS). Suitable for image analysis.
* **Recommended:** NVIDIA GPU with 6GB+ VRAM (CUDA enabled) for real-time video/webcam inference (30+ FPS).

---

## 🧠 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **YOLOv8 for face detection** | Fastest accurate face detector, works on any resolution. |
| **YOLOv8 backbone for classification** | Reuse CSPDarknet features already strong at spatial patterns. |
| **Frozen backbone → gradual unfreeze** | Preserve pretrained features; avoid catastrophic forgetting. |
| **FFT + SRM auxiliary branches** | Force model to learn forensic noise features, not just semantics. |
| **Focal Loss** | Down-weights easy fakes; focuses on hard manipulations. |
| **Dashboard UI (Gradio)** | Unified, professional interface for multi-source forensic analysis. |

---

## 📚 References

* YOLOv8 by Ultralytics: [GitHub Repository](https://github.com/ultralytics/ultralytics)
* FaceForensics++ Dataset: [Rossler et al., 2019](https://arxiv.org/abs/1901.08971)
* Spatial-Rich Model (SRM) for image noise extraction (Fridrich & Kodovsky, 2012).
* Gradio Web Interface Documentation: [Gradio](https://www.gradio.app/)