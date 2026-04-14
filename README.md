# DeepFake Detector — YOLOv8 Pipeline
### Master-Level Computer Vision Project (M2)

> Two-stage real-time deepfake detection:
> **Stage 1** — YOLOv8 face detector (locates every face in the frame)
> **Stage 2** — YOLOv8 backbone + custom classification head (REAL / FAKE per face)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                            │
│                                                                      │
│  Webcam frame                                                        │
│      │                                                               │
│      ▼                                                               │
│  ┌─────────────────┐                                                 │
│  │  YOLOv8 Face    │  Detects face bounding boxes                   │
│  │  Detector       │  (YOLOv8n fine-tuned on WiderFace)             │
│  └────────┬────────┘                                                 │
│           │  List[(x1,y1,x2,y2)]                                     │
│           ▼                                                           │
│  ┌─────────────────┐                                                 │
│  │  Face Crop +    │  Pad · resize to 224×224 · normalize           │
│  │  Preprocess     │                                                 │
│  └────────┬────────┘                                                 │
│           │  Tensor (B, 3, 224, 224)                                 │
│           ▼                                                           │
│  ┌─────────────────────────────────────────────────┐                │
│  │          DeepFake Classification Model           │                │
│  │                                                  │                │
│  │  YOLOv8 Backbone (CSPDarknet, pretrained)       │                │
│  │       ↓                                          │                │
│  │  Feature Pyramid (P3 · P4 · P5)                 │                │
│  │       ↓                                          │                │
│  │  Frequency Branch  +  Texture Branch             │                │
│  │  (FFT noise maps)     (SRM residuals)            │                │
│  │       ↓                                          │                │
│  │  Attention Fusion → Classification Head          │                │
│  │       ↓                                          │                │
│  │  Sigmoid → P(fake) ∈ [0, 1]                     │                │
│  └────────┬────────────────────────────────────────┘                │
│           │                                                           │
│           ▼                                                           │
│  Overlay on frame: bounding box + REAL/FAKE label + confidence bar  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
deepfake_yolo/
├── src/
│   ├── data/
│   │   ├── dataset.py           # Dataset for training classification head
│   │   ├── augmentation.py      # Albumentations pipelines
│   │   └── face_extractor.py    # YOLOv8-based face detector wrapper
│   ├── models/
│   │   ├── backbone.py          # YOLOv8 backbone extractor (feature maps)
│   │   ├── classification_head.py  # Custom deepfake head on top of backbone
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
├── app/
│   ├── webcam_detector.py       # OpenCV live detection app
│   └── streamlit_app.py         # Streamlit web dashboard
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Full test evaluation
│   ├── extract_faces.py         # Batch face extraction from video dataset
│   └── download_weights.py      # Download YOLOv8 pretrained weights
├── configs/
│   └── default.yaml             # All hyperparameters
├── tests/
│   └── test_pipeline.py         # Unit tests
├── docs/
│   └── technical_report.md      # M2-level documentation
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install
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

# 6a. Live webcam detection
python app/webcam_detector.py --checkpoint weights/best_model.pth

# 6b. Streamlit web UI
streamlit run app/streamlit_app.py

# 7. Run unit tests
pytest tests/ -v
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| YOLOv8 for face detection | Fastest accurate face detector, works on any resolution |
| YOLOv8 backbone for classification | Reuse CSPDarknet features already strong at spatial patterns |
| Frozen backbone → gradual unfreeze | Preserve pretrained features; avoid catastrophic forgetting |
| FFT + SRM auxiliary branches | Force model to learn forensic noise features, not just semantics |
| Focal Loss | Down-weights easy fakes; focuses on hard manipulations |
| Temporal smoothing (webcam) | Stable predictions despite per-frame detection jitter |
