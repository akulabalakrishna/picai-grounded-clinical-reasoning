"""
Visualizes model predictions on real cases -- the "is this anatomically
plausible" step of the smoke-test checklist (not blank, not everywhere,
roughly where a lesion would make sense given the GT contour).

Deliberately imports find_dataset_root / load_case_split / preprocess_case /
UNet3D directly from kaggle_train_baseline_unet.py instead of reimplementing
them, so the preprocessing used here is guaranteed identical to what the
model was trained on -- no risk of a second copy silently drifting out of
sync (same reasoning as picai_geom.py being shared between the sanity check
and training).

Usage (on Kaggle, in the same notebook/session as training, or locally
after downloading a checkpoint):
  python visualize_predictions.py --checkpoint /kaggle/working/checkpoints/step_0000123_epoch_0004_end.pt
  python visualize_predictions.py --checkpoint <path> --n-cases 10 --split both

Also prints a quick automated check per case (predicted foreground voxel
fraction vs ground truth) to catch a degenerate "always predicts nothing"
or "predicts everything" model even before you look at the images --
useful because eyeballing 6 images can miss a model that's subtly wrong in
a way that's obvious in the numbers.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from kaggle_train_baseline_unet import (  # noqa: E402
    find_dataset_root, load_case_split, preprocess_case, UNet3D, KAGGLE_WORKING,
)

# NOT Path(__file__).parent -- if this script is attached to a Kaggle
# notebook as a "utility script" rather than pasted into a cell, that
# location mounts read-only (same as an input dataset), and writing there
# fails. KAGGLE_WORKING (/kaggle/working, or the local test path off-Kaggle)
# is the only location guaranteed writable, same as the training script.
OUT_DIR = KAGGLE_WORKING / "prediction_overlays"

# A trained model that predicts foreground on >50% of voxels, or on
# essentially none, is degenerate regardless of what the loss curve says --
# these thresholds are deliberately loose (real lesions are usually a small
# fraction of the volume) and meant to catch gross failure modes, not to be
# a precise quality bar.
DEGENERATE_LOW_FRACTION = 1e-6
DEGENERATE_HIGH_FRACTION = 0.5


def load_model(checkpoint_path: Path, device):
    model = UNet3D().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}  "
          f"(epoch={ckpt['epoch']}, global_step={ckpt['global_step']})")
    return model


def best_slice_index(gt: np.ndarray, pred_prob: np.ndarray) -> int:
    score = gt.sum(axis=(1, 2)) * 10 + (pred_prob > 0.5).sum(axis=(1, 2))
    return int(np.argmax(score)) if score.max() > 0 else gt.shape[0] // 2


def visualize_case(model, root: Path, case_id: str, device, out_dir: Path):
    x, y = preprocess_case(root / "images" / case_id, root / "labels" / case_id, case_id)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x_t)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred_binary = (prob > 0.5).astype(np.float32)

    fg_fraction = float(pred_binary.mean())
    gt_fg_fraction = float(y.mean())
    degenerate = fg_fraction < DEGENERATE_LOW_FRACTION or fg_fraction > DEGENERATE_HIGH_FRACTION
    print(f"  {case_id}: pred_fg_fraction={fg_fraction:.6f}  gt_fg_fraction={gt_fg_fraction:.6f}"
          f"  [{'DEGENERATE' if degenerate else 'ok'}]")

    z = best_slice_index(y, prob)
    t2w_slice = x[0, z]  # channel 0 = t2w, already per-slice z-normed by preprocess_case

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(t2w_slice, cmap="gray")
    axes[0].set_title(f"{case_id}  T2W slice {z}")
    axes[0].axis("off")

    axes[1].imshow(t2w_slice, cmap="gray")
    if y[z].max() > 0:
        axes[1].contour(y[z], colors="lime", linewidths=1.5, levels=[0.5])
    if pred_binary[z].max() > 0:
        axes[1].contour(pred_binary[z], colors="red", linewidths=1.5, levels=[0.5])
    axes[1].set_title("GT lesion (green) vs predicted (red)")
    axes[1].axis("off")

    im = axes[2].imshow(prob[z], cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Predicted probability heatmap")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    fig.tight_layout()
    out_path = out_dir / f"{case_id}_prediction.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return degenerate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint file")
    parser.add_argument("--n-cases", type=int, default=6)
    parser.add_argument("--split", choices=["train", "val", "both"], default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = find_dataset_root()
    train_ids, val_ids = load_case_split(root)
    case_ids = {"train": train_ids, "val": val_ids, "both": train_ids + val_ids}[args.split]
    case_ids = case_ids[: args.n_cases]

    model = load_model(Path(args.checkpoint), device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_degenerate = sum(visualize_case(model, root, cid, device, OUT_DIR) for cid in case_ids)

    print(f"\n{len(case_ids) - n_degenerate}/{len(case_ids)} cases look non-degenerate.")
    if n_degenerate > 0:
        print("WARNING: some predictions are degenerate (blank, or covering >50% of the "
              "volume). Before concluding the model is broken: check how many global_steps "
              "it has actually trained for (an undertrained model often predicts all-background "
              "first, since FOCAL_ALPHA=0.97 heavily weights the rare positive class -- give it "
              "more steps before judging), then check FOCAL_ALPHA/FOCAL_GAMMA and LR.")
    print(f"Overlays: {OUT_DIR}")


if __name__ == "__main__":
    main()
