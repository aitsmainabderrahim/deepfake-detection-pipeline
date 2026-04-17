"""
app/streamlit_app.py — Streamlit web UI for YOLOv8 DeepFake Detector.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile
from pathlib import Path

import cv2
import av
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from streamlit_webrtc import webrtc_streamer

# 1. Page Configuration (تكون هي الأولى)
st.set_page_config(
    page_title="DeepFake Detector (YOLOv8)",
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #080d18; }
h1,h2,h3 { color: #e2e8f0; }
.verdict-real { background:#0d2d1a; border:1px solid #16a34a;
                border-radius:10px; padding:18px; text-align:center; }
.verdict-fake { background:#2d0d0d; border:1px solid #dc2626;
                border-radius:10px; padding:18px; text-align:center; }
</style>""", unsafe_allow_html=True)


# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline(checkpoint: str, config_path: str):
    from src.utils.config import load_config
    from src.models.deepfake_model import build_model
    from src.data.face_extractor import YOLOFaceExtractor

    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = build_model(cfg)
    ckpt   = torch.load(checkpoint, map_location=device)
    
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()

    extractor = YOLOFaceExtractor(
        weights=cfg.face_detector.weights,
        confidence=cfg.face_detector.confidence,
        target_size=cfg.data.image_size,
        device=cfg.face_detector.device,
    )
    return model, cfg, device, extractor


def predict_face(model, cfg, device, face_bgr: np.ndarray) -> float:
    from src.features.frequency import compute_fft_feature_map
    from src.features.texture import compute_texture_feature_map

    rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1).to(device)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1).to(device)
    t    = torch.from_numpy(rgb).permute(2,0,1).float()/255.0
    t    = ((t - mean) / std).unsqueeze(0).to(device)

    fft  = compute_fft_feature_map(rgb, size=224)
    tex  = compute_texture_feature_map(rgb, size=224)
    feat = torch.from_numpy(np.concatenate([fft, tex], axis=0)).unsqueeze(0).to(device)

    with torch.no_grad():
        logit, _ = model(t, feat)
        return torch.sigmoid(logit).item()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.markdown("## ⚙️ Model Config")
    ckpt   = st.sidebar.text_input("Checkpoint", "weights/checkpoints/best_model.pth")
    config = st.sidebar.text_input("Config",     "configs/default.yaml")
    thresh = st.sidebar.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Pipeline:**\n"
        "1. YOLOv8 detects face boxes\n"
        "2. YOLOv8 backbone extracts features\n"
        "3. FFT + SRM forensic branch\n"
        "4. Attention fusion → REAL/FAKE"
    )
    return ckpt, config, thresh


# ── Verdict widget ────────────────────────────────────────────────────────────

def verdict_html(prob: float, threshold: float) -> str:
    is_fake = prob >= threshold
    label   = "FAKE ⚠️" if is_fake else "REAL ✅"
    conf    = prob if is_fake else 1 - prob
    color   = "#ef4444" if is_fake else "#22c55e"
    cls_    = "verdict-fake" if is_fake else "verdict-real"
    bar_w   = prob * 100
    return f"""
    <div class="{cls_}">
      <div style="font-size:2rem;font-weight:900;color:{color};letter-spacing:3px">{label}</div>
      <div style="color:#aaa;margin:6px 0">Confidence: <b style="color:white">{conf*100:.1f}%</b></div>
      <div style="background:#333;border-radius:6px;height:14px;overflow:hidden;margin-top:8px">
        <div style="width:{bar_w:.1f}%;height:100%;background:linear-gradient(90deg,#22c55e,#ef4444)"></div>
      </div>
      <div style="display:flex;justify-content:space-between;color:#666;font-size:.75rem;margin-top:3px">
        <span>REAL</span><span>FAKE</span>
      </div>
    </div>"""


# ── Tab 1: Image ──────────────────────────────────────────────────────────────

def tab_image(model, cfg, device, extractor, threshold):
    st.header("📷 Image Analysis")
    up = st.file_uploader("Upload image", type=["jpg","jpeg","png","webp"])
    if not up:
        st.info("Upload an image to detect and classify faces.")
        return

    data     = np.frombuffer(up.read(), np.uint8)
    img_bgr  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("Stage 1: detecting faces…"):
        faces = extractor.extract_all(img_bgr)

    if not faces:
        st.warning("No face detected. Try a clearer, front-facing photo.")
        st.image(img_rgb, use_container_width=True)
        return

    st.success(f"✓ {len(faces)} face(s) detected")

    for i, (face_crop, box) in enumerate(faces):
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        with st.spinner(f"Stage 2: classifying face {i+1}…"):
            prob = predict_face(model, cfg, device, face_crop)

        st.markdown(f"### Face {i+1}")
        c1, c2, c3 = st.columns([1, 1.5, 1.5])
        with c1:
            annotated = img_rgb.copy()
            x1,y1,x2,y2 = map(int, box)
            color = (220,50,50) if prob >= threshold else (50,220,50)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 3)
            st.image(annotated, caption="Detected box", use_container_width=True)
        with c2:
            st.image(face_rgb, caption="Face crop (224×224)", use_container_width=True)
        with c3:
            st.markdown(verdict_html(prob, threshold), unsafe_allow_html=True)
            st.markdown(f"<br>**Raw P(fake):** `{prob:.4f}`", unsafe_allow_html=True)

        with st.expander(f"Feature maps — Face {i+1}"):
            from src.features.frequency import compute_fft_spectrum
            from src.features.texture import compute_srm, compute_gradient

            face_u8 = face_crop.astype(np.uint8)
            spec    = compute_fft_spectrum(face_u8)
            srm     = compute_srm(face_u8)[0]
            grad    = compute_gradient(face_u8)

            fig, axes = plt.subplots(1, 4, figsize=(14, 3), facecolor="#0d1117")
            for ax in axes: ax.set_facecolor("#0d1117")

            axes[0].imshow(face_rgb)
            axes[0].set_title("Face crop", color="white")
            axes[1].imshow(spec, cmap="inferno")
            axes[1].set_title("FFT spectrum", color="white")
            axes[2].imshow(srm, cmap="RdBu_r")
            axes[2].set_title("SRM residual", color="white")
            axes[3].imshow(grad, cmap="hot")
            axes[3].set_title("Gradient map", color="white")
            for ax in axes: ax.axis("off")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.markdown("---")


# ── Tab 2: Video ──────────────────────────────────────────────────────────────

def tab_video(model, cfg, device, extractor, threshold):
    st.header("🎬 Video Analysis")
    up = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"])
    if not up:
        st.info("Upload a video file.")
        return

    n_frames = st.slider("Frames to analyze", 10, 100, 30)
    if not st.button("▶ Analyze"):
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(up.read())
        vpath = tmp.name

    cap   = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    idxs  = np.linspace(0, total-1, n_frames, dtype=int)

    probs, times = [], []
    prog   = st.progress(0)
    prev   = st.empty()

    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret: continue

        faces = extractor.extract_all(frame)
        if faces:
            p = predict_face(model, cfg, device, faces[0][0])
            probs.append(p)
            times.append(idx / fps)
            
            if i % 8 == 0:
                face_rgb = cv2.cvtColor(faces[0][0], cv2.COLOR_BGR2RGB)
                prev.image(face_rgb, caption=f"t={idx/fps:.1f}s — P(fake)={p:.3f}", width=224)
        
        prog.progress((i+1)/n_frames)

    cap.release()
    if os.path.exists(vpath): os.unlink(vpath)

    if not probs:
        st.error("No faces found in video.")
        return

    probs = np.array(probs)
    mean_p = probs.mean()
    fake_frames_count = np.sum(probs >= threshold)
    fake_ratio = fake_frames_count / len(probs)

    is_final_fake = fake_ratio >= 0.1 
    final_verdict = "FAKE ⚠️" if is_final_fake else "REAL ✅"

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean P(fake)", f"{mean_p:.3f}")
    c2.metric("Frames as FAKE", f"{fake_ratio*100:.1f}%")
    c3.metric("Verdict", final_verdict)

    display_p = max(mean_p, threshold + 0.05) if is_final_fake else min(mean_p, threshold - 0.05)
    st.markdown(verdict_html(display_p, threshold), unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=probs, mode="lines+markers",
                             line=dict(color="#ef4444", width=2),
                             fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
                             name="P(fake)"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="white",
                  annotation_text=f"Threshold ({threshold})")
    fig.update_layout(template="plotly_dark", title="Deepfake probability over time",
                      xaxis_title="Time (s)", yaxis_title="P(fake)",
                      yaxis_range=[0,1])
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Real-Time Live ─────────────────────────────────────────────────────

def tab_live(model, cfg, device, extractor, threshold):
    st.header("🎥 Real-Time Detection")
    st.write("Live analysis - Full Width Mode")

    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            faces = extractor.extract_all(img)
            
            for face_crop, box in faces:
                prob = predict_face(model, cfg, device, face_crop)
                is_fake = prob >= threshold
                color = (0, 0, 255) if is_fake else (0, 255, 0)
                label = f"{'FAKE' if is_fake else 'REAL'} {prob:.2f}"
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="deepfake-live-full-width",
        video_frame_callback=VideoProcessor().recv,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )

    st.info("💡 Tip: Full-screen mode might show more lag depending on your camera resolution.")

# ── Tab 4: About ──────────────────────────────────────────────────────────────

def tab_about():
    st.header("📖 Architecture & Usage")
    st.markdown("""
    ### Two-Stage Pipeline
    **Stage 1 — YOLOv8 Face Detector**
    Detects faces using YOLOv8 fine-tuned on WiderFace.
    **Stage 2 — Forensic Backbone**
    Passes crops through a multi-branch network (RGB + FFT + SRM).
    """)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("🔍 DeepFake Detector — YOLOv8 Pipeline")
    st.markdown("---")

    ckpt, config, threshold = sidebar()

    model = extractor = cfg = device = None
    if Path(ckpt).exists():
        with st.spinner("Loading model…"):
            try:
                model, cfg, device, extractor = load_pipeline(ckpt, config)
                st.sidebar.success(f"✓ Model loaded")
            except Exception as e:
                st.sidebar.error(f"Load failed: {e}")
    else:
        st.warning(f"Checkpoint `{ckpt}` not found.")

    t1, t2, t3, t4 = st.tabs(["📷 Image", "🎬 Video", "🎥 Live", "📖 About"])
    
    with t1:
        if model: tab_image(model, cfg, device, extractor, threshold)
    with t2:
        if model: tab_video(model, cfg, device, extractor, threshold)
    with t3:
        if model: tab_live(model, cfg, device, extractor, threshold)
    with t4:
        tab_about()

if __name__ == "__main__":
    main()