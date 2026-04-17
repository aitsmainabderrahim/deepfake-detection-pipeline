"""
tests/test_pipeline.py

Unit tests for the YOLOv8 DeepFake Detector pipeline.

Coverage:
  - FFT feature computation
  - Texture (SRM, LBP, gradient) features
  - FrequencyBranch CNN
  - DeepFakeHead forward pass
  - Full DeepFakeYOLOModel (mock backbone)
  - Loss functions
  - Metrics
  - Dataset structure
  - End-to-end inference smoke test

Run:
    pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import pytest
import cv2


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def face_bgr():
    """224×224 synthetic BGR face image."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

@pytest.fixture
def face_rgb(face_bgr):
    return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

@pytest.fixture
def batch_rgb():
    return torch.randn(2, 3, 224, 224)

@pytest.fixture
def batch_feat():
    return torch.randn(2, 10, 224, 224)


# ─── FFT Features ────────────────────────────────────────────────────────────

class TestFFTFeatures:
    def test_spectrum_shape(self, face_rgb):
        from src.features.frequency import compute_fft_spectrum
        s = compute_fft_spectrum(face_rgb)
        assert s.shape == (224, 224)
        assert s.dtype == np.float32

    def test_spectrum_range(self, face_rgb):
        from src.features.frequency import compute_fft_spectrum
        s = compute_fft_spectrum(face_rgb, log_scale=True)
        assert s.min() >= 0.0
        assert s.max() <= 1.0 + 1e-6

    def test_high_pass_shape(self, face_rgb):
        from src.features.frequency import compute_high_pass
        hp = compute_high_pass(face_rgb)
        assert hp.shape == face_rgb.shape
        assert hp.dtype == np.float32

    def test_feature_map_shape(self, face_rgb):
        from src.features.frequency import compute_fft_feature_map
        f = compute_fft_feature_map(face_rgb, size=224)
        assert f.shape == (5, 224, 224)
        assert f.dtype == np.float32

    def test_tensor_output(self, face_rgb):
        from src.features.frequency import fft_tensor
        t = fft_tensor(face_rgb, size=224)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (5, 224, 224)


# ─── Texture Features ─────────────────────────────────────────────────────────

class TestTextureFeatures:
    def test_srm_shape(self, face_bgr):
        from src.features.texture import compute_srm
        s = compute_srm(face_bgr)
        assert s.shape == (3, 224, 224)
        assert s.dtype == np.float32

    def test_srm_range(self, face_bgr):
        from src.features.texture import compute_srm
        s = compute_srm(face_bgr)
        assert s.min() >= -1.0 - 1e-6
        assert s.max() <=  1.0 + 1e-6

    def test_lbp_shape(self, face_bgr):
        from src.features.texture import compute_lbp
        lbp = compute_lbp(face_bgr)
        assert lbp.shape == (224, 224)
        assert lbp.dtype == np.float32

    def test_gradient_shape(self, face_bgr):
        from src.features.texture import compute_gradient
        g = compute_gradient(face_bgr)
        assert g.shape == (224, 224)
        assert 0.0 <= g.min() and g.max() <= 1.0 + 1e-6

    def test_texture_map_shape(self, face_bgr):
        from src.features.texture import compute_texture_feature_map
        t = compute_texture_feature_map(face_bgr, size=224)
        assert t.shape == (5, 224, 224)

    def test_texture_tensor(self, face_rgb):
        from src.features.texture import texture_tensor
        t = texture_tensor(face_rgb, size=224)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (5, 224, 224)

    def test_combined_channels(self, face_rgb):
        """FFT (5ch) + texture (5ch) = 10 channels."""
        from src.features.frequency import fft_tensor
        from src.features.texture import texture_tensor
        fft = fft_tensor(face_rgb)
        tex = texture_tensor(face_rgb)
        combined = torch.cat([fft, tex], dim=0)
        assert combined.shape == (10, 224, 224)


# ─── Model Components ─────────────────────────────────────────────────────────

class TestFrequencyBranch:
    @pytest.fixture
    def branch(self):
        from src.models.classification_head import FrequencyBranch
        return FrequencyBranch(in_channels=10, out_dim=256)

    def test_output_shape(self, branch, batch_feat):
        out = branch(batch_feat)
        assert out.shape == (2, 256)

    def test_output_is_finite(self, branch, batch_feat):
        out = branch(batch_feat)
        assert torch.isfinite(out).all()

    def test_different_batch_sizes(self, branch):
        from src.models.classification_head import FrequencyBranch
        b = FrequencyBranch(10, 256)
        for bs in [1, 4, 8]:
            x   = torch.randn(bs, 10, 224, 224)
            out = b(x)
            assert out.shape == (bs, 256)


class TestCrossAttentionFusion:
    def test_output_shape(self):
        from src.models.classification_head import CrossAttentionFusion
        gate = CrossAttentionFusion(dim=256)
        f1   = torch.randn(4, 256)
        f2   = torch.randn(4, 256)
        out  = gate(f1, f2)
        assert out.shape == (4, 256)

    def test_weights_sum_to_one(self):
        from src.models.classification_head import CrossAttentionFusion
        gate = CrossAttentionFusion(dim=64)
        # Access gate weights directly
        f1, f2 = torch.randn(2, 64), torch.randn(2, 64)
        combined = torch.cat([f1, f2], dim=-1)
        w = gate.gate(combined)
        # Softmax → sums to 1
        assert torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-5)


class TestDeepFakeHead:
    @pytest.fixture
    def head(self):
        from src.models.classification_head import DeepFakeHead
        return DeepFakeHead(
            backbone_dim=128, feat_channels=10,
            hidden_dim=64, dropout=0.1,
            use_freq=True, use_attention=True, use_aux_head=True,
        )

    def test_forward_with_features(self, head, batch_feat):
        head.train()
        backbone_feat = torch.randn(2, 128)
        logit, aux    = head(backbone_feat, batch_feat)
        assert logit.shape == (2, 1)
        assert aux.shape   == (2, 1)

    def test_forward_without_features(self, head):
        head.eval()
        backbone_feat = torch.randn(2, 128)
        logit, aux    = head(backbone_feat, None)
        assert logit.shape == (2, 1)
        assert aux is None

    def test_no_aux_in_eval(self, head, batch_feat):
        head.eval()
        backbone_feat = torch.randn(2, 128)
        _, aux = head(backbone_feat, batch_feat)
        # aux_head only fires during training
        assert aux is None


class TestDeepFakeYOLOModel:
    """
    Tests the full model using a mocked backbone to avoid
    needing real YOLOv8 weights during unit testing.
    """

    @pytest.fixture
    def model_with_mock_backbone(self):
        """Build model, then replace backbone with a simple mock."""
        from src.models.deepfake_model import DeepFakeYOLOModel
        from src.models.backbone import YOLOv8BackboneExtractor

        # Patch backbone to a simple linear layer
        class MockBackbone(torch.nn.Module):
            out_dim = 128
            def forward(self, x):
                B = x.shape[0]
                return torch.randn(B, 128)
            def freeze(self, freeze=True): pass

        # Build model (may fail to load yolo weights → uses fallback)
        m = DeepFakeYOLOModel.__new__(DeepFakeYOLOModel)
        torch.nn.Module.__init__(m)
        m.backbone = MockBackbone()

        from src.models.classification_head import DeepFakeHead
        m.head = DeepFakeHead(
            backbone_dim=128, feat_channels=10,
            hidden_dim=64, dropout=0.1,
        )
        return m

    def test_forward_returns_logit(self, model_with_mock_backbone, batch_rgb):
        model_with_mock_backbone.eval()
        logit, aux = model_with_mock_backbone(batch_rgb, None)
        assert logit.shape == (2, 1)

    def test_forward_with_feat(self, model_with_mock_backbone,
                               batch_rgb, batch_feat):
        model_with_mock_backbone.train()
        logit, aux = model_with_mock_backbone(batch_rgb, batch_feat)
        assert logit.shape == (2, 1)
        assert aux.shape   == (2, 1)

    def test_predict_proba_range(self, model_with_mock_backbone, batch_rgb):
        probs = model_with_mock_backbone.predict_proba(batch_rgb)
        assert probs.shape == (2,)
        assert (probs >= 0).all() and (probs <= 1).all()


# ─── Losses ───────────────────────────────────────────────────────────────────

class TestLosses:
    def test_focal_positive(self):
        from src.models.losses import FocalLoss
        fn     = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 1)
        labels = torch.randint(0, 2, (8,)).float()
        loss   = fn(logits, labels)
        assert loss.item() > 0

    def test_label_smooth_positive(self):
        from src.models.losses import LabelSmoothBCE
        fn     = LabelSmoothBCE(smoothing=0.1)
        logits = torch.randn(8, 1)
        labels = torch.randint(0, 2, (8,)).float()
        loss   = fn(logits, labels)
        assert loss.item() > 0

    def test_combined_with_aux(self):
        from src.models.losses import CombinedLoss
        fn     = CombinedLoss(primary="focal", aux_weight=0.3)
        logits = torch.randn(8, 1)
        aux    = torch.randn(8, 1)
        labels = torch.randint(0, 2, (8,)).float()
        loss   = fn(logits, labels, aux)
        assert loss.item() > 0

    def test_combined_without_aux(self):
        from src.models.losses import CombinedLoss
        fn     = CombinedLoss()
        logits = torch.randn(8, 1)
        labels = torch.randint(0, 2, (8,)).float()
        loss   = fn(logits, labels, aux_logit=None)
        assert loss.item() > 0

    def test_focal_vs_bce_ordering(self):
        """Focal loss should be lower than BCE for easy examples (high-conf correct preds)."""
        from src.models.losses import FocalLoss, LabelSmoothBCE
        focal = FocalLoss(gamma=2.0, alpha=0.5)
        bce   = LabelSmoothBCE(smoothing=0.0)
        # Perfect predictions (easy examples)
        logits = torch.tensor([[10.0], [10.0], [-10.0], [-10.0]])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        assert focal(logits, labels).item() < bce(logits, labels).item()


# ─── Metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_predictions(self):
        from src.utils.metrics import compute_metrics
        labels = np.array([0, 0, 1, 1])
        probs  = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(labels, probs)
        assert m["acc"] == pytest.approx(1.0)
        assert m["auc"] == pytest.approx(1.0)
        assert m["f1"]  == pytest.approx(1.0)

    def test_all_keys_present(self):
        from src.utils.metrics import compute_metrics
        m = compute_metrics(np.array([0,1,0,1]), np.array([0.2,0.8,0.3,0.7]))
        for k in ["acc","precision","recall","f1","auc","ap","eer"]:
            assert k in m

    def test_eer_between_zero_and_one(self):
        from src.utils.metrics import compute_metrics
        np.random.seed(0)
        labels = np.random.randint(0, 2, 100)
        probs  = np.random.rand(100)
        m = compute_metrics(labels, probs)
        assert 0.0 <= m["eer"] <= 1.0


# ─── End-to-End Smoke Test ────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_inference_pipeline(self, face_rgb):
        """
        Complete pipeline without real YOLOv8 weights:
          synthetic face → FFT+texture features → mock model → probability
        """
        from src.features.frequency import fft_tensor
        from src.features.texture import texture_tensor
        from src.models.classification_head import DeepFakeHead

        # Build minimal model
        class MockBackbone(torch.nn.Module):
            out_dim = 128
            def forward(self, x): return torch.randn(x.shape[0], 128)

        backbone = MockBackbone()
        head     = DeepFakeHead(
            backbone_dim=128, feat_channels=10, hidden_dim=64,
            use_freq=True, use_attention=True, use_aux_head=False,
        )

        # Prepare inputs
        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        t    = torch.from_numpy(face_rgb).permute(2,0,1).float() / 255.0
        t    = ((t - mean) / std).unsqueeze(0)

        fft  = fft_tensor(face_rgb).unsqueeze(0)
        tex  = texture_tensor(face_rgb).unsqueeze(0)
        feat = torch.cat([fft, tex], dim=1)   # (1, 10, 224, 224)

        # Forward
        backbone.eval(); head.eval()
        with torch.no_grad():
            bb_feat      = backbone(t)
            logit, aux   = head(bb_feat, feat)
            prob         = torch.sigmoid(logit).item()

        assert 0.0 <= prob <= 1.0
        assert aux is None   # eval mode, use_aux_head=False

    def test_feature_pipeline_deterministic(self, face_rgb):
        """Same input → same feature tensor (no randomness in feature extraction)."""
        from src.features.frequency import fft_tensor
        from src.features.texture import texture_tensor

        f1 = fft_tensor(face_rgb)
        f2 = fft_tensor(face_rgb)
        assert torch.allclose(f1, f2)

        t1 = texture_tensor(face_rgb)
        t2 = texture_tensor(face_rgb)
        assert torch.allclose(t1, t2)
