"""
Direction D, inference-only against the completed baseline checkpoint.
No training, no gradients -- pure forward passes, meant for one short GPU
job (or a slow CPU run if that's all that's available). Combines all
three of today's Direction D asks into one script/pass so cases are only
preprocessed and run through the model once each, reused across whichever
of the three analyses need that case:

  1. CALIBRATION: per-voxel Expected Calibration Error (ECE) of the
     baseline's sigmoid lesion probabilities against real ground truth,
     accumulated over the held-out val set (standard practice -- held-out
     cases, not train cases the model was fit to).

  2. ROBUSTNESS: Dice degradation under three inference-time-only
     perturbations (Gaussian noise, Gaussian blur, constant intensity
     shift), applied ONLY to the t2w/adc/hbv channels -- gland/zone are
     precomputed anatomical-prior channels (outputs of a separate upstream
     segmentation step), not acquired images, so "imaging robustness"
     doesn't apply to them. Perturbations are applied to the already
     per-slice-normalized model input (the exact tensor the network
     consumes), at fixed, documented magnitudes -- not exposed as tunable
     knobs, so results are reproducible without a parameter sweep.

  3. MASK-RATIONALE CONSISTENCY: compares concepts derived from the
     model's OWN predicted mask (lesion_present, zone_location, shape)
     against the concepts already recorded in each case's rationale JSON
     (which were derived from the GROUND-TRUTH mask by
     kaggle_rationale_synthesis.py). Run over every case with a rationale
     on disk (train+val both -- this measures prediction/rationale
     agreement, not generalization, so the split doesn't matter here).

     SCOPE NOTE, stated explicitly rather than silently fudged:
     t2w_intensity_zscore is NOT compared. The rationale's z-score is
     computed against RAW, native-resolution T2W intensity in native
     physical space (kaggle_rationale_synthesis.py's own load_aligned_case);
     the model's input t2w channel is per-slice z-normalized in a
     resampled/cropped/resized model space (train_direction_b.py's
     preprocess_case). Reconciling the two would require resampling the
     predicted mask back onto the native T2W grid -- real work, not done
     today. The three concepts below (presence/zone/shape) don't have
     this problem: they're computed directly from the predicted mask's
     own geometry, in the model's own space, using that space's own
     correctly-derived effective voxel spacing (MODEL_SPACING_ZYX below)
     -- valid without needing the native-space reconstruction.

sphericity() and the size/shape thresholds below are copied VERBATIM from
kaggle_rationale_synthesis.py (not imported -- that module runs a `pip
install` subprocess unconditionally at import time, the wrong side effect
to trigger from here), so the SAME geometric definitions used to write
the rationales are used to re-derive concepts from predictions.

Usage:
  python direction_d_inference.py \\
      --data-root ~/picai_data \\
      --rationales-dir ~/picai_data/rationales \\
      --baseline-checkpoint ~/picai_outputs/run1/checkpoints/step_0005040_epoch_0059_end.pt \\
      --output-dir ~/picai_direction_d_outputs \\
      --n-robustness-cases 20
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_direction_b import (
    UNet3D, preprocess_case, load_case_split, autocast_ctx,
    dice_on_lesion_positive_slices, TARGET_SPACING, CROP_SHAPE_ZYX, FINAL_SHAPE_ZYX,
)

try:
    from skimage.measure import marching_cubes, mesh_surface_area
except ImportError:
    marching_cubes = None

# ---- copied verbatim from kaggle_rationale_synthesis.py (see docstring) ----
PZ_LABEL, TZ_LABEL = 1, 2
LARGE_LESION_MM3 = 1500.0
IRREGULAR_SPHERICITY = 0.6


def sphericity(mask_arr, spacing_zyx):
    volume_mm3 = float(mask_arr.sum() * np.prod(spacing_zyx))
    if volume_mm3 <= 0 or marching_cubes is None:
        return None
    padded = np.pad(mask_arr.astype(np.uint8), 1)
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing_zyx)
    except (RuntimeError, ValueError):
        return None
    area_mm2 = mesh_surface_area(verts, faces)
    if area_mm2 <= 0:
        return None
    return float((36 * np.pi * volume_mm3 ** 2) ** (1 / 3) / area_mm2)


# Effective voxel spacing of the MODEL's own input/output space: TARGET_SPACING
# survives resample_to_spacing and center_crop_or_pad unchanged (crop/pad don't
# resample), then resize_zyx upsamples Y,X from CROP_SHAPE to FINAL_SHAPE
# (Z is never resized -- F.interpolate's size=(Z,...) with matching Z asserted
# in resize_zyx), shrinking effective spacing proportionally. TARGET_SPACING is
# (x,y,z) (SimpleITK convention); CROP_SHAPE_ZYX/FINAL_SHAPE_ZYX are (z,y,x).
MODEL_SPACING_ZYX = (
    TARGET_SPACING[2] * (CROP_SHAPE_ZYX[0] / FINAL_SHAPE_ZYX[0]),  # z
    TARGET_SPACING[1] * (CROP_SHAPE_ZYX[1] / FINAL_SHAPE_ZYX[1]),  # y
    TARGET_SPACING[0] * (CROP_SHAPE_ZYX[2] / FINAL_SHAPE_ZYX[2]),  # x
)
MODEL_VOXEL_VOL_MM3 = float(np.prod(MODEL_SPACING_ZYX))


def zone_location_from_mask(lesion_mask: np.ndarray, zone_arr: np.ndarray) -> str:
    """Same 0.65-fraction rule as kaggle_rationale_synthesis.py's synthesize_rationale."""
    zone_rounded = np.round(zone_arr)
    pz = int((lesion_mask & (zone_rounded == PZ_LABEL)).sum())
    tz = int((lesion_mask & (zone_rounded == TZ_LABEL)).sum())
    n_zoned = pz + tz
    if n_zoned == 0:
        return "mixed"
    pz_frac, tz_frac = pz / n_zoned, tz / n_zoned
    if pz_frac >= 0.65:
        return "PZ"
    if tz_frac >= 0.65:
        return "TZ"
    return "mixed"


def predicted_concepts(pred_mask: np.ndarray, zone_arr: np.ndarray):
    """pred_mask, zone_arr: (Z,Y,X) arrays in MODEL space (FINAL_SHAPE_ZYX)."""
    lesion_present = bool(pred_mask.any())
    if not lesion_present:
        return {"lesion_present": False, "zone_location": "none",
                "shape_irregular": None, "lesion_volume_cc": 0.0}
    volume_mm3 = float(pred_mask.sum() * MODEL_VOXEL_VOL_MM3)
    zone_loc = zone_location_from_mask(pred_mask, zone_arr)
    sph = sphericity(pred_mask, MODEL_SPACING_ZYX)
    shape_irregular = (sph is not None and sph < IRREGULAR_SPHERICITY)
    return {"lesion_present": True, "zone_location": zone_loc,
            "shape_irregular": shape_irregular, "lesion_volume_cc": volume_mm3 / 1000.0}


# ============================ PERTURBATIONS =================================
# Applied only to channels 0,1,2 (t2w, adc, hbv); channels 3,4 (gland, zone)
# are precomputed label priors, not acquired images, and are left untouched.
# Fixed, documented magnitudes -- not swept -- applied to the already
# per-slice-normalized model input.

def _gaussian_kernel3d(sigma: float, device):
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    kz = g1d.view(-1, 1, 1)
    ky = g1d.view(1, -1, 1)
    kx = g1d.view(1, 1, -1)
    kernel3d = kz * ky * kx
    return kernel3d, radius


def apply_perturbation(x: torch.Tensor, kind: str) -> torch.Tensor:
    """x: (1, 5, Z, Y, X). Returns a perturbed copy; channels 3,4 untouched."""
    x = x.clone()
    img_channels = x[:, :3]
    if kind == "gaussian_noise":
        img_channels = img_channels + torch.randn_like(img_channels) * 0.25
    elif kind == "gaussian_blur":
        kernel3d, radius = _gaussian_kernel3d(sigma=1.5, device=x.device)
        kernel = kernel3d.view(1, 1, *kernel3d.shape).repeat(3, 1, 1, 1, 1)
        img_channels = F.conv3d(img_channels, kernel, padding=radius, groups=3)
    elif kind == "intensity_shift":
        img_channels = img_channels + 0.5
    else:
        raise ValueError(kind)
    x[:, :3] = img_channels
    return x


# ============================ CALIBRATION ====================================

class ECEAccumulator:
    """Standard binary ECE: confidence = P(predicted class), accuracy = whether
    predicted class matches ground truth, both binned by confidence. Accumulated
    incrementally per case (not storing all voxels) since a single case is
    ~25M voxels."""

    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.bin_count = np.zeros(n_bins, dtype=np.float64)
        self.bin_conf_sum = np.zeros(n_bins, dtype=np.float64)
        self.bin_correct_sum = np.zeros(n_bins, dtype=np.float64)

    def update(self, probs: np.ndarray, target: np.ndarray):
        pred_pos = probs >= 0.5
        confidence = np.where(pred_pos, probs, 1 - probs)
        correct = (pred_pos == (target >= 0.5)).astype(np.float64)
        bin_idx = np.minimum((confidence * self.n_bins).astype(np.int64), self.n_bins - 1)
        for b in range(self.n_bins):
            m = bin_idx == b
            self.bin_count[b] += m.sum()
            self.bin_conf_sum[b] += confidence[m].sum()
            self.bin_correct_sum[b] += correct[m].sum()

    def result(self):
        total = self.bin_count.sum()
        ece = 0.0
        per_bin = []
        for b in range(self.n_bins):
            if self.bin_count[b] == 0:
                per_bin.append({"bin": b, "count": 0, "avg_confidence": None, "accuracy": None})
                continue
            avg_conf = self.bin_conf_sum[b] / self.bin_count[b]
            acc = self.bin_correct_sum[b] / self.bin_count[b]
            ece += (self.bin_count[b] / total) * abs(acc - avg_conf)
            per_bin.append({"bin": b, "count": int(self.bin_count[b]),
                             "avg_confidence": float(avg_conf), "accuracy": float(acc)})
        return {"ece": float(ece), "total_voxels": int(total), "per_bin": per_bin}


# ================================= MAIN =====================================


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--rationales-dir", type=Path, required=True)
    p.add_argument("--baseline-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-robustness-cases", type=int, default=20,
                   help="How many val cases to run the 3 perturbations on (4x the "
                        "clean-inference cost per case included -- kept smaller than "
                        "the full val set by default to bound runtime).")
    p.add_argument("--ece-bins", type=int, default=15)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    unet = UNet3D().to(device).eval()
    ckpt = torch.load(args.baseline_checkpoint, map_location=device)
    unet.load_state_dict(ckpt["model_state"])
    print(f"Loaded baseline checkpoint: epoch={ckpt['epoch']} global_step={ckpt['global_step']}", flush=True)

    train_ids, val_ids = load_case_split(args.data_root)
    print(f"train={len(train_ids)} val={len(val_ids)}", flush=True)

    images_dir, labels_dir = args.data_root / "images", args.data_root / "labels"

    # ---------------- 1. CALIBRATION (val set, clean images) ----------------
    ece_acc = ECEAccumulator(n_bins=args.ece_bins)
    clean_dice_by_case = {}
    t0 = time.time()
    for i, case_id in enumerate(val_ids):
        x, y = preprocess_case(images_dir / case_id, labels_dir / case_id, case_id)
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        yt = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad(), autocast_ctx(device.type):
            logits, _ = unet(xt)
        probs = torch.sigmoid(logits.float()).cpu().numpy()[0, 0]
        target = y
        ece_acc.update(probs, target)
        d = dice_on_lesion_positive_slices(logits, yt)
        if d is not None:
            clean_dice_by_case[case_id] = d
        print(f"  [calibration {i+1}/{len(val_ids)}] {case_id}  clean_dice={d}  "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

    ece_result = ece_acc.result()
    print(f"CALIBRATION DONE: ECE={ece_result['ece']:.4f} over {ece_result['total_voxels']:,} voxels, "
          f"{len(val_ids)} val cases, mean_clean_dice={np.mean(list(clean_dice_by_case.values())):.4f}", flush=True)
    with open(args.output_dir / "calibration_result.json", "w") as f:
        json.dump({"ece_result": ece_result, "clean_dice_by_case": clean_dice_by_case,
                   "n_val_cases": len(val_ids)}, f, indent=2)

    # ---------------- 2. ROBUSTNESS (subset of val set) ----------------
    robustness_cases = val_ids[:args.n_robustness_cases]
    robustness_results = []
    for i, case_id in enumerate(robustness_cases):
        x, y = preprocess_case(images_dir / case_id, labels_dir / case_id, case_id)
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        yt = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)
        case_result = {"case_id": case_id, "clean_dice": clean_dice_by_case.get(case_id)}
        for kind in ("gaussian_noise", "gaussian_blur", "intensity_shift"):
            xp = apply_perturbation(xt, kind)
            with torch.no_grad(), autocast_ctx(device.type):
                logits, _ = unet(xp)
            d = dice_on_lesion_positive_slices(logits, yt)
            case_result[f"dice_{kind}"] = d
        robustness_results.append(case_result)
        print(f"  [robustness {i+1}/{len(robustness_cases)}] {case_id}  {case_result}  "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

    def _mean(key):
        vals = [r[key] for r in robustness_results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    robustness_summary = {
        "n_cases": len(robustness_cases),
        "mean_clean_dice": _mean("clean_dice"),
        "mean_dice_gaussian_noise": _mean("dice_gaussian_noise"),
        "mean_dice_gaussian_blur": _mean("dice_gaussian_blur"),
        "mean_dice_intensity_shift": _mean("dice_intensity_shift"),
    }
    print(f"ROBUSTNESS DONE: {robustness_summary}", flush=True)
    with open(args.output_dir / "robustness_result.json", "w") as f:
        json.dump({"summary": robustness_summary, "per_case": robustness_results}, f, indent=2)

    # ---------------- 3. MASK-RATIONALE CONSISTENCY (all cases with a rationale) ----------------
    all_case_ids = sorted(p.stem for p in args.rationales_dir.glob("*.json"))
    consistency_results = []
    for i, case_id in enumerate(all_case_ids):
        rationale_path = args.rationales_dir / f"{case_id}.json"
        rationale = json.loads(rationale_path.read_text())
        try:
            x, y = preprocess_case(images_dir / case_id, labels_dir / case_id, case_id)
        except FileNotFoundError as e:
            print(f"  [consistency {i+1}/{len(all_case_ids)}] {case_id}  SKIPPED (missing image/label): {e}", flush=True)
            continue
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad(), autocast_ctx(device.type):
            logits, _ = unet(xt)
        pred_mask = (torch.sigmoid(logits.float()) > 0.5).cpu().numpy()[0, 0].astype(bool)
        zone_arr = x[4]  # model-space zone channel

        pred = predicted_concepts(pred_mask, zone_arr)
        mismatches = []
        if pred["lesion_present"] != rationale["lesion_present"]:
            mismatches.append("lesion_present")
        if rationale["lesion_present"] and pred["lesion_present"]:
            if pred["zone_location"] != rationale["zone_location"]:
                mismatches.append("zone_location")
            rationale_irregular = (rationale["shape_sphericity"] is not None
                                    and rationale["shape_sphericity"] < IRREGULAR_SPHERICITY)
            if pred["shape_irregular"] is not None and pred["shape_irregular"] != rationale_irregular:
                mismatches.append("shape")
        consistency_results.append({
            "case_id": case_id, "rationale_lesion_present": rationale["lesion_present"],
            "rationale_zone": rationale.get("zone_location"),
            "rationale_shape_sphericity": rationale.get("shape_sphericity"),
            "predicted": pred, "mismatches": mismatches,
        })
        print(f"  [consistency {i+1}/{len(all_case_ids)}] {case_id}  mismatches={mismatches}  "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

    n_total = len(consistency_results)
    n_any_mismatch = sum(1 for r in consistency_results if r["mismatches"])
    consistency_summary = {
        "n_cases": n_total,
        "n_any_mismatch": n_any_mismatch,
        "mismatch_rate": n_any_mismatch / n_total if n_total else None,
        "n_lesion_present_mismatch": sum(1 for r in consistency_results if "lesion_present" in r["mismatches"]),
        "n_zone_mismatch": sum(1 for r in consistency_results if "zone_location" in r["mismatches"]),
        "n_shape_mismatch": sum(1 for r in consistency_results if "shape" in r["mismatches"]),
    }
    print(f"CONSISTENCY DONE: {consistency_summary}", flush=True)
    with open(args.output_dir / "consistency_result.json", "w") as f:
        json.dump({"summary": consistency_summary, "per_case": consistency_results}, f, indent=2)

    print(f"\nAll done. Total elapsed: {time.time()-t0:.1f}s. Results in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
