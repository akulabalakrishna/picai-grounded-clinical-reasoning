"""
SINGLE-CELL, SELF-CONTAINED Kaggle script: Direction D, inference-only
against the completed baseline checkpoint (step_0005040_epoch_0059_end.pt,
60/60 epochs, trained on the IITH cluster's V100 -- see results_summary.md).
No training, no gradients -- pure forward passes. Adapted from
direction_d_inference.py (cluster/argparse version) following the exact
single-cell pattern of kaggle_single_cell_train.py: pip-bootstrap at the
top, find_dataset_root() searching /kaggle/input, GPU-adaptive torch
handling, then the actual work in one continuous script/log.

Combines all three Direction D analyses in one pass so each case is only
preprocessed and forward-passed once, reused across whichever analyses
need it:
  1. CALIBRATION: per-voxel ECE of the baseline's sigmoid probabilities
     against ground truth, over the held-out val set.
  2. ROBUSTNESS: Dice degradation under 3 inference-time-only perturbations
     (gaussian_noise, gaussian_blur, intensity_shift) applied to the
     t2w/adc/hbv channels only (gland/zone are precomputed anatomical
     priors, not acquired images).
  3. MASK-RATIONALE CONSISTENCY: predicted-mask-derived concepts
     (lesion_present, zone_location, shape) vs. the concepts already
     recorded in each case's rationale JSON (derived from the GROUND-TRUTH
     mask). Run over every case with a rationale on disk. t2w_intensity_
     zscore is NOT compared -- see direction_d_inference.py's docstring for
     why (native-space vs. model-space z-score mismatch, not reconciled).

GPU-ADAPTIVE TORCH HANDLING (this is the part that differs from a plain
copy of direction_d_inference.py, and from kaggle_single_cell_train.py's
unconditional pin): kaggle_single_cell_train.py pins torch==2.4.1
unconditionally because every one of 5 real training runs got assigned a
P100 (compute capability 6.0/sm_60), which Kaggle's default preinstalled
torch (compiled for capability 7.0+) can't run on. This script instead
queries nvidia-smi BEFORE importing torch at all, and only applies that
same pin if a P100 is actually the assigned GPU -- if Kaggle hands out a
T4 (capability 7.5) or anything newer, Kaggle's preinstalled torch already
supports it, so this script leaves it alone rather than needlessly
downgrading. Either way, print_environment_diagnostics() below reports the
actual torch version and GPU in the log so this is never a silent choice.

sphericity() and the size/shape thresholds are copied VERBATIM from
kaggle_rationale_synthesis.py / direction_d_inference.py (not imported --
that module runs a `pip install` subprocess unconditionally at import
time, the wrong side effect to trigger from here).
"""
import subprocess
import sys


def _detect_gpu_names():
    """Queries nvidia-smi directly, BEFORE torch is even installed/imported,
    so the pip-install decision below can react to the real assigned GPU
    instead of guessing. Returns [] if nvidia-smi isn't available (no GPU,
    or a non-Kaggle/CPU-only environment) -- never raises."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    except Exception as e:
        print(f"[startup] nvidia-smi GPU probe failed ({e}) -- assuming no legacy-GPU compat issue.")
    return []


_gpu_names_pre_torch = _detect_gpu_names()
print(f"[startup] nvidia-smi GPU probe (pre-torch-install): {_gpu_names_pre_torch or '(none detected)'}", flush=True)
_NEEDS_LEGACY_TORCH_PIN = any("P100" in name for name in _gpu_names_pre_torch)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "SimpleITK", "scikit-image"],
    check=True,
)

if _NEEDS_LEGACY_TORCH_PIN:
    # See module docstring: Kaggle's default preinstalled torch (confirmed via
    # 5 real failed runs on 2026-09-07: torch 2.10.0+cu128) narrowed its
    # prebuilt wheel's compiled kernel images to compute capability 7.0+,
    # which does NOT include sm_60 (Pascal / P100). torch==2.4.1 is the most
    # recent release still shipping sm_60 support in its official wheels.
    print("[startup] P100 detected -- pinning torch==2.4.1 (last release with sm_60 support). "
          "See kaggle_single_cell_train.py's docstring for the failed-run history behind this.", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
         "torch==2.4.1", "--extra-index-url", "https://download.pytorch.org/whl/cu121"],
        check=True,
    )
else:
    print("[startup] No P100 detected (T4, other modern GPU, or no GPU at all) -- keeping "
          "Kaggle's preinstalled torch build rather than downgrading unnecessarily.", flush=True)

import json
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from skimage.measure import marching_cubes, mesh_surface_area
except ImportError:
    marching_cubes = None


def autocast_ctx(device_type: str):
    try:
        return torch.amp.autocast(device_type, enabled=(device_type == "cuda"))
    except TypeError:
        return torch.cuda.amp.autocast(enabled=(device_type == "cuda"))


# ============================== CONFIG ======================================
# Config knobs you might edit between runs.

N_ROBUSTNESS_CASES = 20   # how many val cases get the 3 perturbations (4x cost/case)
ECE_BINS = 15
BASELINE_CHECKPOINT_NAME = "step_0005040_epoch_0059_end.pt"  # exact filename expected

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR = KAGGLE_WORKING if KAGGLE_INPUT_ROOT.exists() else (
    Path(__file__).resolve().parent / "_direction_d_local_test_output"
)

# Overrides -- set if auto-detection below picks the wrong path.
EXPLICIT_DATASET_ROOT = None
EXPLICIT_CHECKPOINT_PATH = None

# ============================ DATA LOCATION =================================
# Identical search pattern to kaggle_single_cell_train.py's find_dataset_root().


def find_dataset_root() -> Path:
    if EXPLICIT_DATASET_ROOT is not None:
        if not (EXPLICIT_DATASET_ROOT / "images").exists():
            raise FileNotFoundError(f"EXPLICIT_DATASET_ROOT={EXPLICIT_DATASET_ROOT} has no 'images' subfolder.")
        return EXPLICIT_DATASET_ROOT

    if KAGGLE_INPUT_ROOT.exists():
        for images_dir in KAGGLE_INPUT_ROOT.rglob("images"):
            if images_dir.is_dir() and (images_dir.parent / "labels").exists():
                return images_dir.parent
        import zipfile
        for zip_path in KAGGLE_INPUT_ROOT.rglob("*.zip"):
            extract_to = KAGGLE_WORKING / f"extracted_{zip_path.stem}"
            if not extract_to.exists():
                extract_to.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extract_to)
            if (extract_to / "images").exists():
                return extract_to
            for images_dir in extract_to.rglob("images"):
                if (images_dir.parent / "labels").exists():
                    return images_dir.parent

    # off-Kaggle local test path -- small 10-case fixture with its own
    # selected_cases.csv/images/labels/rationales, used to verify the script
    # runs end-to-end before it goes anywhere near Kaggle.
    local = Path(r"E:\Cancer_IITH\picai\_direction_b_local_test")
    if (local / "images").exists():
        return local
    raise FileNotFoundError(
        "Could not find the picai dataset under /kaggle/input or locally. "
        "Did you add the Kaggle Dataset to this notebook? If mounted somewhere "
        "unexpected, set EXPLICIT_DATASET_ROOT above."
    )


def _global_step_from_name(path: Path) -> int:
    # step_0000123_epoch_0004_end.pt -> 123
    return int(path.stem.split("_")[1])


def find_baseline_checkpoint(dataset_root: Path) -> Path:
    """Searches for the exact baseline checkpoint filename first (the real
    60/60-epoch, V100-trained baseline this analysis is supposed to run
    against -- see results_summary.md). Falls back to the highest-global-step
    step_*_epoch_*_*.pt found, but prints a loud warning if it has to fall
    back, so a substitute checkpoint is never used silently."""
    if EXPLICIT_CHECKPOINT_PATH is not None:
        if not EXPLICIT_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"EXPLICIT_CHECKPOINT_PATH={EXPLICIT_CHECKPOINT_PATH} does not exist.")
        return EXPLICIT_CHECKPOINT_PATH

    search_roots = [KAGGLE_INPUT_ROOT, dataset_root, KAGGLE_WORKING]
    for root in search_roots:
        if not root.exists():
            continue
        exact = list(root.rglob(BASELINE_CHECKPOINT_NAME))
        if exact:
            return exact[0]

    candidates = []
    for root in search_roots:
        if root.exists():
            candidates += list(root.rglob("step_*_epoch_*_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found anywhere under {search_roots}. Expected "
            f"{BASELINE_CHECKPOINT_NAME} (or at least a step_*_epoch_*_*.pt file) "
            f"to be attached as a Kaggle Dataset to this kernel."
        )
    best = max(candidates, key=_global_step_from_name)
    print(f"WARNING: exact baseline checkpoint {BASELINE_CHECKPOINT_NAME} not found -- "
          f"falling back to highest-global-step checkpoint found instead: {best}. "
          f"Results below are NOT against the intended baseline unless this is it.", flush=True)
    return best


# ============================ GEOMETRY / PREPROCESSING ======================
# Verbatim from train_direction_b.py / kaggle_single_cell_train.py -- exact
# fidelity required for the checkpoint's learned weights to see the same
# input distribution they were trained on.

TARGET_SPACING = (0.5, 0.5, 3.0)          # (x, y, z) mm
CROP_SHAPE_ZYX = (24, 384, 384)           # (z, y, x)
FINAL_SHAPE_ZYX = (24, 1024, 1024)        # (z, y, x)
N_INPUT_CHANNELS = 5                       # t2w, adc, hbv, gland, zone
SEED = 0


def _find_label_path(labels_dir: Path, case_id: str, kind: str) -> Path:
    base = labels_dir / case_id / f"{case_id}_{kind}"
    for ext in (".nii.gz", ".nii"):
        candidate = base.with_name(base.name + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {kind} label found for case {case_id} in {labels_dir / case_id}")


def _resample_to_reference(image, reference, is_label):
    return sitk.Resample(
        image, reference, sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        0, image.GetPixelID(),
    )


def load_aligned_case(images_dir: Path, labels_dir: Path, case_id: str):
    t2w = sitk.ReadImage(str(images_dir / case_id / f"{case_id}_t2w.mha"))
    adc = sitk.ReadImage(str(images_dir / case_id / f"{case_id}_adc.mha"))
    hbv = sitk.ReadImage(str(images_dir / case_id / f"{case_id}_hbv.mha"))
    gland = sitk.ReadImage(str(_find_label_path(labels_dir, case_id, "gland")))
    zone = sitk.ReadImage(str(_find_label_path(labels_dir, case_id, "zone")))
    lesion = sitk.ReadImage(str(_find_label_path(labels_dir, case_id, "lesion")))
    return {
        "t2w": t2w,
        "adc": _resample_to_reference(adc, t2w, is_label=False),
        "hbv": _resample_to_reference(hbv, t2w, is_label=False),
        "gland": _resample_to_reference(sitk.Cast(gland, sitk.sitkUInt8), t2w, is_label=True),
        "zone": _resample_to_reference(sitk.Cast(zone, sitk.sitkUInt8), t2w, is_label=True),
        "lesion": _resample_to_reference(sitk.Cast(lesion, sitk.sitkUInt8), t2w, is_label=True),
    }


def resample_to_spacing(image: sitk.Image, target_spacing, is_label: bool) -> sitk.Image:
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [max(1, int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))) for i in range(3)]
    return sitk.Resample(
        image, new_size, sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        image.GetOrigin(), target_spacing, image.GetDirection(), 0, image.GetPixelID(),
    )


def center_crop_or_pad(arr: np.ndarray, target_zyx) -> np.ndarray:
    out = arr
    for axis, target in enumerate(target_zyx):
        cur = out.shape[axis]
        if cur == target:
            continue
        if cur > target:
            start = (cur - target) // 2
            out = np.take(out, range(start, start + target), axis=axis)
        else:
            pad_total = target - cur
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_width = [(0, 0)] * out.ndim
            pad_width[axis] = (pad_before, pad_after)
            out = np.pad(out, pad_width, mode="constant", constant_values=0)
    return out


def resize_zyx(arr: np.ndarray, target_zyx, is_label: bool) -> np.ndarray:
    assert arr.shape[0] == target_zyx[0]
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    mode = "nearest" if is_label else "trilinear"
    kwargs = {} if is_label else {"align_corners": False}
    resized = F.interpolate(t, size=target_zyx, mode=mode, **kwargs)
    return resized.squeeze(0).squeeze(0).numpy()


def per_slice_znorm(volume_zyx: np.ndarray) -> np.ndarray:
    out = np.empty_like(volume_zyx, dtype=np.float32)
    for z in range(volume_zyx.shape[0]):
        sl = volume_zyx[z]
        m, s = sl.mean(), sl.std()
        out[z] = (sl - m) / s if s > 1e-6 else sl - m
    return out


def preprocess_case(case_dir_images: Path, case_dir_labels: Path, case_id: str):
    aligned = load_aligned_case(case_dir_images.parent, case_dir_labels.parent, case_id)
    resampled = {n: resample_to_spacing(im, TARGET_SPACING, is_label=n in ("gland", "zone", "lesion"))
                 for n, im in aligned.items()}
    arrs = {n: sitk.GetArrayFromImage(im) for n, im in resampled.items()}
    cropped = {n: center_crop_or_pad(a, CROP_SHAPE_ZYX) for n, a in arrs.items()}
    resized = {n: resize_zyx(a, FINAL_SHAPE_ZYX, is_label=n in ("gland", "zone", "lesion"))
               for n, a in cropped.items()}
    t2w = per_slice_znorm(resized["t2w"].astype(np.float32))
    adc = per_slice_znorm(resized["adc"].astype(np.float32))
    hbv = per_slice_znorm(resized["hbv"].astype(np.float32))
    gland = (resized["gland"] > 0.5).astype(np.float32)
    zone = resized["zone"].astype(np.float32)
    lesion = (resized["lesion"] > 0.5).astype(np.float32)
    input_5ch = np.stack([t2w, adc, hbv, gland, zone], axis=0)
    return input_5ch.astype(np.float32), lesion.astype(np.float32)


def load_case_split(root: Path):
    import csv
    with open(root / "selected_cases.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    positives = [r["case_id"] for r in rows if r["case_csPCa"] == "YES"]
    negatives = [r["case_id"] for r in rows if r["case_csPCa"] == "NO"]
    import random
    random.Random(SEED).shuffle(positives)
    random.Random(SEED).shuffle(negatives)
    n_val_pos = max(1, len(positives) // 5)
    n_val_neg = max(1, len(negatives) // 5)
    val_ids = positives[:n_val_pos] + negatives[:n_val_neg]
    train_ids = positives[n_val_pos:] + negatives[n_val_neg:]
    return train_ids, val_ids


# ================================= MODEL =====================================
# Identical to the baseline's UNet3D -- required for the checkpoint's
# state_dict to load (torch.load(...).load_state_dict(...) fails loudly on
# any mismatch; it will NOT silently load wrong weights).


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.InstanceNorm3d(out_ch, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.InstanceNorm3d(out_ch, affine=True), nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return checkpoint(self.net, x, use_reentrant=False)


class UNet3D(nn.Module):
    def __init__(self, in_channels=N_INPUT_CHANNELS, base_ch=16):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.enc1 = ConvBlock(in_channels, chs[0])
        self.enc2 = ConvBlock(chs[0], chs[1])
        self.enc3 = ConvBlock(chs[1], chs[2])
        self.bottleneck = ConvBlock(chs[2], chs[3])
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.up3 = nn.ConvTranspose3d(chs[3], chs[2], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = ConvBlock(chs[2] * 2, chs[2])
        self.up2 = nn.ConvTranspose3d(chs[2], chs[1], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = ConvBlock(chs[1] * 2, chs[1])
        self.up1 = nn.ConvTranspose3d(chs[1], chs[0], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlock(chs[0] * 2, chs[0])
        self.out_conv = nn.Conv3d(chs[0], 1, kernel_size=1)
        self.bottleneck_channels = chs[3]

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.out_conv(d1)
        return logits, b


def dice_on_lesion_positive_slices(pred_logits, target):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    dices = []
    for b in range(target.shape[0]):
        for z in range(target.shape[2]):
            t_slice = target[b, 0, z]
            if t_slice.sum() == 0:
                continue
            p_slice = pred[b, 0, z]
            intersection = (p_slice * t_slice).sum()
            denom = p_slice.sum() + t_slice.sum()
            dices.append((2 * intersection / denom).item() if denom > 0 else 1.0)
    return float(np.mean(dices)) if dices else None


# ---- copied verbatim from kaggle_rationale_synthesis.py / direction_d_inference.py ----
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


# Effective voxel spacing of the MODEL's own input/output space -- see
# direction_d_inference.py's docstring for the derivation.
MODEL_SPACING_ZYX = (
    TARGET_SPACING[2] * (CROP_SHAPE_ZYX[0] / FINAL_SHAPE_ZYX[0]),
    TARGET_SPACING[1] * (CROP_SHAPE_ZYX[1] / FINAL_SHAPE_ZYX[1]),
    TARGET_SPACING[0] * (CROP_SHAPE_ZYX[2] / FINAL_SHAPE_ZYX[2]),
)
MODEL_VOXEL_VOL_MM3 = float(np.prod(MODEL_SPACING_ZYX))


def zone_location_from_mask(lesion_mask: np.ndarray, zone_arr: np.ndarray) -> str:
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


def _gaussian_kernel3d(sigma: float, device):
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    kz = g1d.view(-1, 1, 1)
    ky = g1d.view(1, -1, 1)
    kx = g1d.view(1, 1, -1)
    return kz * ky * kx, radius


def apply_perturbation(x: torch.Tensor, kind: str) -> torch.Tensor:
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


# ============================== DIAGNOSTICS ==================================


def print_environment_diagnostics(device):
    import platform
    print(f"Python: {platform.python_version()}")
    print(f"Torch:  {torch.__version__}   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"GPU(s): {n_gpu}")
        for i in range(n_gpu):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name}  {props.total_memory / 1e9:.1f} GB  "
                  f"compute_capability={props.major}.{props.minor}")
    print(f"SimpleITK: {sitk.Version_VersionString()}")
    if device.type != "cuda":
        print("WARNING: no GPU detected -- inference will be slow at "
              f"{FINAL_SHAPE_ZYX} resolution.")


# ================================= MAIN =====================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print_environment_diagnostics(device)

    root = find_dataset_root()
    print(f"Dataset root: {root}", flush=True)
    rationales_dir = root / "rationales"

    ckpt_path = find_baseline_checkpoint(root)
    print(f"Baseline checkpoint: {ckpt_path}", flush=True)

    unet = UNet3D().to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt["model_state"])
    print(f"Loaded baseline checkpoint: epoch={ckpt['epoch']} global_step={ckpt['global_step']}", flush=True)

    train_ids, val_ids = load_case_split(root)
    print(f"train={len(train_ids)} val={len(val_ids)}", flush=True)

    images_dir, labels_dir = root / "images", root / "labels"

    # ---------------- 1. CALIBRATION (val set, clean images) ----------------
    ece_acc = ECEAccumulator(n_bins=ECE_BINS)
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
    mean_clean_dice = float(np.mean(list(clean_dice_by_case.values()))) if clean_dice_by_case else None
    print(f"CALIBRATION DONE: ECE={ece_result['ece']:.4f} over {ece_result['total_voxels']:,} voxels, "
          f"{len(val_ids)} val cases, mean_clean_dice={mean_clean_dice}", flush=True)
    with open(OUTPUT_DIR / "calibration_result.json", "w") as f:
        json.dump({"ece_result": ece_result, "clean_dice_by_case": clean_dice_by_case,
                   "n_val_cases": len(val_ids)}, f, indent=2)

    # ---------------- 2. ROBUSTNESS (subset of val set) ----------------
    robustness_cases = val_ids[:N_ROBUSTNESS_CASES]
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
    with open(OUTPUT_DIR / "robustness_result.json", "w") as f:
        json.dump({"summary": robustness_summary, "per_case": robustness_results}, f, indent=2)

    # ---------------- 3. MASK-RATIONALE CONSISTENCY (all cases with a rationale) ----------------
    all_case_ids = sorted(p.stem for p in rationales_dir.glob("*.json")) if rationales_dir.exists() else []
    consistency_results = []
    for i, case_id in enumerate(all_case_ids):
        rationale_path = rationales_dir / f"{case_id}.json"
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
        zone_arr = x[4]

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
    with open(OUTPUT_DIR / "consistency_result.json", "w") as f:
        json.dump({"summary": consistency_summary, "per_case": consistency_results}, f, indent=2)

    print(f"\nAll done. Total elapsed: {time.time()-t0:.1f}s. Results in {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
