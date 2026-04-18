import sys, os
import cv2
import torch
import numpy as np
import gradio as gr
from pathlib import Path

# Add project root to sys.path to resolve imports
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.config import load_config
from src.models.deepfake_model import build_model
from src.data.face_extractor import YOLOFaceExtractor
from src.features.frequency import compute_fft_feature_map
from src.features.texture import compute_texture_feature_map

# --- 1. Pipeline Loader ---
def load_pipeline():
    config_path = "configs/default.yaml"
    checkpoint  = "weights/checkpoints/best_model.pth"
    
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = build_model(cfg)
    
    # Load weights safely
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()
    
    # Initialize YOLOv8 face extractor
    extractor = YOLOFaceExtractor(
        weights=cfg.face_detector.weights,
        confidence=cfg.face_detector.confidence,
        target_size=cfg.data.image_size,
        device=cfg.face_detector.device,
    )
    return model, cfg, device, extractor

# Load the pipeline globally on startup
try:
    model, cfg, device, extractor = load_pipeline()
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")

# --- 2. Image & Webcam Inference (Multi-Face) ---
def predict_gradio(img, threshold=0.5):
    if img is None: 
        return None, None, "Please provide an input."
    
    # Fix readonly array error and ensure uint8 for skimage
    img = np.ascontiguousarray(img).copy().astype(np.uint8)
    
    # Convert RGB to BGR for extractor
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = extractor.extract_all(img_bgr)
    
    if not faces:
        return img, {"FAKE": 0.0, "REAL": 0.0}, "No faces detected."

    final_report = ""
    res_label = {"FAKE": 0.0, "REAL": 0.0}
    
    for i, (face_crop_bgr, box) in enumerate(faces):
        face_rgb = cv2.cvtColor(face_crop_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        
        # Normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        t = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        t = ((t - mean) / std).unsqueeze(0).to(device)

        # Feature Extraction
        fft_feat = compute_fft_feature_map(face_rgb, size=224)
        tex_feat = compute_texture_feature_map(face_rgb, size=224)
        feat = torch.from_numpy(np.concatenate([fft_feat, tex_feat], axis=0)).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            logit, _ = model(t, feat)
            prob = torch.sigmoid(logit).item()

        # Verdict and Drawing
        is_fake = prob >= threshold
        label = "FAKE" if is_fake else "REAL"
        color = (255, 0, 0) if is_fake else (0, 255, 0) 
        
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img, f"Face {i+1}: {label} ({prob:.2f})", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        final_report += f"Face {i+1}: {label} | "
        
        # Keep the highest probabilities for the label UI
        if is_fake:
            res_label["FAKE"] = max(res_label["FAKE"], float(prob))
        else:
            res_label["REAL"] = max(res_label["REAL"], float(1-prob))

    return img, res_label, final_report

# --- 3. Video Inference ---
def analyze_video_gradio(video_path, threshold=0.5):
    if not video_path:
        return None, "Please upload a video."
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return None, "Invalid video file."
        
    # Sample frames to optimize processing time
    n_frames = 20 
    idxs = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    
    probs = []
    best_fake_frame = None
    max_fake_prob = -1
    
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret: continue
        
        faces = extractor.extract_all(frame)
        if faces:
            face_crop_bgr, box = faces[0]
            face_rgb = cv2.cvtColor(face_crop_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
            
            # Normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            t = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
            t = ((t - mean) / std).unsqueeze(0).to(device)

            # Features
            fft_feat = compute_fft_feature_map(face_rgb, size=224)
            tex_feat = compute_texture_feature_map(face_rgb, size=224)
            feat = torch.from_numpy(np.concatenate([fft_feat, tex_feat], axis=0)).unsqueeze(0).to(device)

            with torch.no_grad():
                logit, _ = model(t, feat)
                prob = torch.sigmoid(logit).item()
                
            probs.append(prob)
            
            # Retain the most suspicious frame
            if prob > max_fake_prob:
                max_fake_prob = prob
                best_fake_frame = frame.copy()
                is_fake = prob >= threshold
                color = (0, 0, 255) if is_fake else (0, 255, 0)
                label = "FAKE" if is_fake else "REAL"
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(best_fake_frame, (x1, y1), (x2, y2), color, 4)
                cv2.putText(best_fake_frame, f"{label} ({prob:.2f})", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
    cap.release()
    
    if not probs:
        return None, "No faces detected in the video."
        
    # Aggregate results
    fake_ratio = sum(p >= threshold for p in probs) / len(probs)
    verdict = "FAKE VIDEO ⚠️" if fake_ratio > 0.1 else "REAL VIDEO ✅"
    
    summary = f"Final Verdict: {verdict}\n" \
              f"Mean P(fake): {np.mean(probs):.2f}\n" \
              f"Frames classified as Fake: {fake_ratio*100:.1f}%"
    
    if best_fake_frame is not None:
        best_fake_frame = cv2.cvtColor(best_fake_frame, cv2.COLOR_BGR2RGB)
        
    return best_fake_frame, summary

# --- 4. Gradio Interface Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ DeepFake Detection System")
    gr.Markdown("Analyzing spatial and frequency artifacts using YOLOv8 and Attention Fusion.")
    
    with gr.Tabs():
        # Tab 1: Image
        with gr.TabItem("📷 Image Analysis"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="numpy", label="Upload Photo")
                    threshold_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Threshold")
                    btn_img = gr.Button("🔍 Analyze Image", variant="primary")
                with gr.Column():
                    output_img = gr.Image(label="Detection Result")
                    label_out = gr.Label(num_top_classes=2, label="Classification")
                    text_out = gr.Textbox(label="Summary")
            
            btn_img.click(predict_gradio, 
                          inputs=[input_img, threshold_slider], 
                          outputs=[output_img, label_out, text_out])

        # Tab 2: Video
        with gr.TabItem("🎬 Video Analysis"):
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(label="Upload Video File")
                    vid_threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Threshold")
                    vid_btn = gr.Button("🎬 Analyze Video", variant="primary")
                with gr.Column():
                    out_vid_frame = gr.Image(label="Most Suspicious Frame")
                    vid_summary = gr.Textbox(label="Analysis Summary", lines=4)
            
            vid_btn.click(analyze_video_gradio, 
                          inputs=[input_video, vid_threshold], 
                          outputs=[out_vid_frame, vid_summary])

        # Tab 3: Webcam
        with gr.TabItem("🎥 Live Webcam"):
            gr.Markdown("### Real-Time Detection Mode")
            gr.Interface(
                fn=predict_gradio,
                inputs=[
                    gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam Feed"),
                    gr.Slider(0.1, 0.9, value=0.5, label="Sensitivity")
                ],
                outputs=[
                    gr.Image(label="Live Results"), 
                    gr.Label(label="Classification Probabilities"), 
                    gr.Textbox(label="Detection Log")
                ],
                live=True
            )

    gr.Markdown("---")
    gr.Markdown("Developed for Deepfake Detection Research.")

# --- 5. Application Launch ---
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(), 
        show_error=True
    )