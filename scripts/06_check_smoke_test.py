"""
Automates as much of the 4-point smoke-test checklist as can be checked
from artifacts alone (training_history.json + the checkpoint directory).
The 4th point (anatomically plausible predictions) still needs your eyes on
the images from visualize_predictions.py -- this script only catches the
numeric proxy for it (predicted foreground fraction), not "does this look
like a real lesion location".

Run this AFTER downloading training_history.json and the checkpoints
folder from Kaggle (or point it at them in place on Kaggle).

Usage:
  python 06_check_smoke_test.py --history /path/to/training_history.json --checkpoints /path/to/checkpoints
"""
import argparse
import json
from pathlib import Path

import torch


def check_completed_without_crashing(history):
    ok = len(history) > 0
    print(f"[1] Training completed epochs without crashing: "
          f"{'PASS' if ok else 'FAIL'} ({len(history)} epoch(s) logged)")
    if not ok:
        print("    -> No epochs logged at all. Check the Kaggle notebook's error output --")
        print("       likely an exception before the first epoch finished (missing package,")
        print("       dataset not found, OOM on the very first forward pass).")
    return ok


def check_checkpoint_resume_evidence(checkpoint_dir: Path):
    files = sorted(checkpoint_dir.glob("step_*_epoch_*_*.pt")) if checkpoint_dir.exists() else []
    print(f"[2] Checkpoint files present: {'PASS' if files else 'FAIL'} ({len(files)} found)")
    if not files:
        print(f"    -> No checkpoints found in {checkpoint_dir}. If training ran, checkpoints")
        print("       should exist -- check CHECKPOINT_DIR / /kaggle/working permissions.")
        return False

    latest = max(files, key=lambda p: int(p.stem.split("_")[1]))
    try:
        ckpt = torch.load(latest, map_location="cpu")
        required_keys = {"epoch", "global_step", "mid_epoch", "model_state",
                          "optimizer_state", "scheduler_state", "scaler_state", "history"}
        missing = required_keys - set(ckpt.keys())
        loadable = not missing
        print(f"    Latest checkpoint {latest.name} loads and has all required keys: "
              f"{'PASS' if loadable else 'FAIL'}")
        if missing:
            print(f"    -> Missing keys: {missing}")
        return loadable
    except Exception as e:
        print(f"    -> FAILED to load {latest}: {e}")
        return False


def check_loss_decreasing(history):
    if len(history) < 4:
        print("[3] Loss trend: SKIPPED (need >=4 epochs to compare meaningfully; "
              f"only {len(history)} logged -- re-check after a longer run)")
        return None
    losses = [h["train_loss"] for h in history]
    n = max(1, len(losses) // 4)
    early_mean = sum(losses[:n]) / n
    late_mean = sum(losses[-n:]) / n
    decreasing = late_mean < early_mean
    print(f"[3] Loss decreasing: {'PASS' if decreasing else 'FAIL'} "
          f"(early mean={early_mean:.4f}, late mean={late_mean:.4f})")
    if not decreasing:
        print("    -> Loss flat or rising suggests gradients aren't flowing usefully: check LR,")
        print("       that FOCAL_ALPHA/FOCAL_GAMMA aren't zeroing the loss, and that the target")
        print("       tensor actually contains positive lesion voxels for at least some cases")
        print("       (an all-negative training batch by bad luck would look like this too --")
        print("       check load_case_split() actually put positives in both train and val).")
    return decreasing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    args = parser.parse_args()

    with open(args.history) as f:
        history = json.load(f)

    print("=== Smoke-test checklist (points 1-3; point 4 needs visualize_predictions.py) ===\n")
    r1 = check_completed_without_crashing(history)
    r2 = check_checkpoint_resume_evidence(args.checkpoints)
    r3 = check_loss_decreasing(history)

    print("\n=== Summary ===")
    print(f"  1. Completed without crashing:        {'PASS' if r1 else 'FAIL'}")
    print(f"  2. Checkpoint structurally valid:      {'PASS' if r2 else 'FAIL'}")
    print(f"  3. Loss decreasing:                    "
          f"{'PASS' if r3 else ('FAIL' if r3 is False else 'SKIPPED')}")
    print("  4. Predictions anatomically plausible:  run visualize_predictions.py and look")
    print("     (this script only checks foreground-fraction isn't degenerate, not location)")
    print("\nNote: this is a pipeline validity check, not a final model-quality result --")
    print("don't read the Dice numbers in the history as meaningful until you've scaled up")
    print("and the scaling-math conversation has set a real epoch/session budget.")


if __name__ == "__main__":
    main()
