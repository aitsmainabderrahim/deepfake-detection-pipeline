"""scripts/evaluate.py — Full test-set evaluation with metrics + plots."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse, json, logging
import numpy as np, torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.models.deepfake_model import build_model
from src.data.dataset import DeepFakeDataset
from src.data.augmentation import get_val_transform
from src.utils.metrics import compute_metrics, print_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--save_dir",   default="results/")
    args = p.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    val_tf  = get_val_transform(cfg.data.image_size)
    test_ds = DeepFakeDataset(cfg.paths.faces_dir, val_tf, split="test")
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            x, labels = batch[:2]
            x = x.to(device); labels = labels.to(device)
            logit, _ = model(x)
            probs_all.extend(torch.sigmoid(logit).squeeze().cpu().numpy().tolist())
            labels_all.extend(labels.cpu().numpy().tolist())

    labels = np.array(labels_all)
    probs  = np.array(probs_all)
    m      = compute_metrics(labels, probs, cfg.evaluation.threshold)
    print_metrics(m, "Test")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.save_dir}/test_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    logger.info(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
