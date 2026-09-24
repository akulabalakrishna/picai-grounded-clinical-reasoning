"""
SINGLE-CELL, SELF-CONTAINED Kaggle script for picai-strict-201pos-220neg.
Paste this whole file into ONE notebook cell and run it -- one paste,
total. No separate !pip install cell, no separate validation cells: when
SIMULATE_INTERRUPT_AFTER_STEPS is None and training completes (full
N_EPOCHS or a clean MAX_TRAIN_HOURS stop), this SAME script automatically
continues in the same process into the smoke-test checks and the
visualization step (see POST-TRAINING VALIDATION section below) -- one
continuous log, no extra cells to paste after a committed run finishes.

When SIMULATE_INTERRUPT_AFTER_STEPS is set (the interrupt test), the script
exits via SystemExit before ever reaching the post-training validation --
nothing runs there in that mode, since there's nothing meaningful to check.

============================================================================
>>> THE ONE LINE YOU EDIT BETWEEN RUNS IS RIGHT BELOW <<<
============================================================================
"""
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "SimpleITK"],
    check=True,
)

# Interactive-editor interrupt/resume test: abandoned as a Kaggle-side
# verification step (two attempts both landed in Save Version/commit mode
# instead of live interactive execution -- a Kaggle UI navigation issue,
# not a code bug). The resume mechanism itself is already verified
# correct: tested locally twice, before and after the merge, with
# byte-identical global_step/loss/dice numbers both times.
#
# Set to None for a real run (via Save Version / commit). Only set this
# back to an int (e.g. 3) if you specifically want to re-attempt the
# interrupt test later -- not needed for the current plan.
SIMULATE_INTERRUPT_AFTER_STEPS = None
# ============================================================================

"""
Kaggle-notebook-ready training script: baseline 3D U-Net for csPCa lesion
segmentation, conditioned on anatomical-prior masks, matching the paper's
Appendix D spec as closely as a free Kaggle GPU allows.

============================================================================
ASSUMPTION FLAGGED (please confirm against your Appendix D text):
  Input to the U-Net = 5 channels: t2w, adc, hbv, gland, zone.
  Target/label        = lesion mask (binary segmentation, Dice-scored).
  Your original message said the loader loads "3-channel input (T2W/ADC/HBV)
  and 3 mask channels (gland/zone/lesion)" -- if the lesion mask is meant to
  ALSO be a 6th input channel (e.g. a self-supervised auxiliary input) rather
  than the sole prediction target, change N_INPUT_CHANNELS below to 6 and
  concatenate `lesion` into the input stack in PicaiDataset.__getitem__.
  As written, this is a standard anatomical-prior-conditioned lesion
  segmentation setup (matches the "rationale-grounded tumor detection"
  framing -- gland/zone are the priors, lesion is what's being detected).
============================================================================

KAGGLE-FORCED DEVIATIONS FROM THE APPENDIX D SPEC (flagged, not silent):

  1. RESOLUTION / MEMORY: a 3D volume at 1024x1024x24 is enormous for a
     16GB T4/P100. This script defaults to batch_size=1, automatic mixed
     precision, gradient checkpointing on every U-Net block, and gradient
     accumulation. If you hit CUDA OOM, set DOWNSCALE_FACTOR=2 below
     (trains at 512x512 instead) -- a real spec deviation, flagged not silent.

  2. SESSION TIME LIMIT: checkpoints every CHECKPOINT_EVERY_N_STEPS
     optimizer steps (mid-epoch, not just at epoch boundaries) AND tracks
     wall-clock time, stopping cleanly before MAX_TRAIN_HOURS (8.5h). Auto-
     resumes from the latest checkpoint found under /kaggle/working or
     /kaggle/input (no path editing needed for a re-uploaded prior run).

  3. DISK: /kaggle/working has a ~20GB quota. This 421-case pack is ~4.4GB
     locally (Kaggle's own copy of the label masks is larger -- see the
     GEOMETRY section below for why -- but still nowhere near the quota).

  4. SAMPLE SIZE: 421 cases (201 real human-expert-verified positive + 220
     negative -- STRICT policy, no AI-derived masks; 337 train / 84 val) is
     the first real baseline run, not a final result. 19 of the original
     220 real-verified positives (8.6%) failed mask-alignment verification
     and were excluded before packaging -- see findings.md in the dataset.

  5. SINGLE GPU ONLY: this run intentionally uses cuda:0 only, even if
     Kaggle assigns a dual-GPU "T4 x2" session -- print_environment_
     diagnostics() below will say so, but no DataParallel/multi-GPU code
     is used here. That's a deliberate choice for this validation run, to
     be revisited only at the multi-fold CV stage.
"""
import json
import math
import random
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset


def make_grad_scaler(device_type: str):
    """torch.amp.GradScaler(device_type, ...) requires torch>=2.3ish; older
    torch (possibly what a given Kaggle image ships, unverified at the time
    this was written) only has torch.cuda.amp.GradScaler(...) with no device
    arg. Kaggle's exact pre-installed torch version isn't something this
    script can check in advance -- print it at startup (see main()) and rely
    on this fallback rather than assuming."""
    try:
        return torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=(device_type == "cuda"))


def autocast_ctx(device_type: str):
    try:
        return torch.amp.autocast(device_type, enabled=(device_type == "cuda"))
    except TypeError:
        return torch.cuda.amp.autocast(enabled=(device_type == "cuda"))


# ============================== CONFIG ====================================

TARGET_SPACING = (0.5, 0.5, 3.0)          # (x, y, z) mm -- Appendix D spec
CROP_SHAPE_ZYX = (24, 384, 384)           # center crop/pad, (z, y, x)
FINAL_SHAPE_ZYX = (24, 1024, 1024)        # final resize, (z, y, x)
DOWNSCALE_FACTOR = 1                      # set to 2 if you hit CUDA OOM (see flag #1)
FINAL_SHAPE_ZYX = (
    FINAL_SHAPE_ZYX[0],
    FINAL_SHAPE_ZYX[1] // DOWNSCALE_FACTOR,
    FINAL_SHAPE_ZYX[2] // DOWNSCALE_FACTOR,
)

N_INPUT_CHANNELS = 5   # t2w, adc, hbv, gland, zone -- see ASSUMPTION flag above
BATCH_SIZE = 1
ACCUM_STEPS = 4        # effective batch size 4
N_EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-3        # Appendix D spec
LR_EXP_GAMMA = 0.97         # exponential decay per epoch (not specified exactly
                             # by the user -- tune against your Appendix D text)
FOCAL_ALPHA = 0.97
FOCAL_GAMMA = 2.0
ROTATION_CHOICES_DEG = [-15, -10, -5, 0, 5, 10, 15]
MAX_TRAIN_HOURS = 8.5
SEED = 0

NUM_WORKERS = 2  # Kaggle's DataLoader worker sandboxing occasionally hangs
                 # with multiprocessing workers -- if the first epoch never
                 # starts, set this to 0 and re-run before suspecting a data bug.
CHECKPOINT_EVERY_N_STEPS = 20   # mid-epoch checkpoint cadence (optimizer steps,
                                # i.e. after ACCUM_STEPS micro-batches each). Keeps
                                # a Kaggle session kill from losing more than ~20
                                # steps of progress even mid-epoch.
CHECKPOINT_KEEP_LAST_N = 3      # rolling retention so 20GB /kaggle/working quota
                                # isn't silently exhausted over a long multi-epoch run
# SIMULATE_INTERRUPT_AFTER_STEPS is defined at the very top of this file --
# that's the one line you edit between runs, not here.

# Post-training validation (only runs when SIMULATE_INTERRUPT_AFTER_STEPS is
# None and training reaches the end of main() normally -- see POST-TRAINING
# VALIDATION section below for why the interrupt-test path never gets here).
VIZ_N_CASES = 6
VIZ_SPLIT = "val"  # "train", "val", or "both"
DEGENERATE_LOW_FRACTION = 1e-6   # predicted foreground below this = suspiciously blank
DEGENERATE_HIGH_FRACTION = 0.5   # predicted foreground above this = suspiciously everywhere

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
CHECKPOINT_DIR = KAGGLE_WORKING / "checkpoints"

# --- Configurable overrides: set these if auto-detection below picks the
# wrong path (e.g. Kaggle mounted your dataset under an unexpected slug, or
# you want to point at a specific prior run's checkpoints explicitly rather
# than relying on the auto-search over /kaggle/input). Leave as None to use
# the auto-detection in find_dataset_root() / find_latest_checkpoint(). ---
EXPLICIT_DATASET_ROOT = None            # e.g. Path("/kaggle/input/picai-sample-30cases")
EXPLICIT_CHECKPOINT_INPUT_DIR = None    # e.g. Path("/kaggle/input/picai-checkpoints-run1")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================ DATA LOCATION ================================


def find_dataset_root() -> Path:
    """
    Locates the picai_sample_421cases folder under /kaggle/input. Does NOT
    assume a specific dataset slug (Kaggle picks the mount folder name from
    your dataset title, which you don't control precisely) -- instead
    searches for an "images" dir with a sibling "labels" dir, at any depth,
    since:
      - uploading via `kaggle datasets create -p kaggle_upload/` (the folder,
        not the zip) mounts the files directly at
        /kaggle/input/<slug>/images/... -- found at depth 1.
      - uploading the zip via the web UI may or may not auto-extract it, and
        if it does, some upload paths nest it one level deeper
        (/kaggle/input/<slug>/picai_sample_421cases/images/...) -- found by
        the recursive search.
      - if Kaggle left it as a raw .zip file instead, this extracts it to
        /kaggle/working (the only writable path) and searches the result.
    Falls back to a local path for testing off-Kaggle.
    """
    if EXPLICIT_DATASET_ROOT is not None:
        if not (EXPLICIT_DATASET_ROOT / "images").exists():
            raise FileNotFoundError(
                f"EXPLICIT_DATASET_ROOT={EXPLICIT_DATASET_ROOT} has no 'images' "
                f"subfolder -- check the path or set it back to None."
            )
        return EXPLICIT_DATASET_ROOT

    if KAGGLE_INPUT_ROOT.exists():
        for images_dir in KAGGLE_INPUT_ROOT.rglob("images"):
            if images_dir.is_dir() and (images_dir.parent / "labels").exists():
                return images_dir.parent

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

    # off-Kaggle local test path
    local = Path(r"E:\Cancer_IITH\picai\kaggle_upload")
    if (local / "images").exists():
        return local
    raise FileNotFoundError(
        "Could not find the picai dataset under /kaggle/input or locally. "
        "Did you add the Kaggle Dataset to this notebook? If it's mounted "
        "somewhere unexpected, set EXPLICIT_DATASET_ROOT above."
    )


# ============================ GEOMETRY / ALIGNMENT ==========================
#
# Ground truth about this dataset's geometry (confirmed by inspection):
#   - t2w, gland, and lesion share the exact same size/spacing/origin/
#     direction as packaged. zone shares spacing but is cropped to a
#     smaller bounding box. adc/hbv are natively lower in-plane resolution.
#     All non-t2w channels are resampled onto the t2w grid below regardless
#     (idempotent where already aligned, required where not).
#
#   - Label file extension: shipped locally as .nii.gz, but Kaggle's
#     dataset ingestion was found (by directly listing the uploaded
#     dataset's files via `kaggle datasets files`, not assumed) to silently
#     decompress gzip'd files uploaded inside a zip-mode directory upload --
#     labels/<case>/<case>_gland.nii.gz came back as ..._gland.nii on
#     Kaggle, ~3x larger, uncompressed. _find_label_path() below tries
#     both extensions so this works either way.


def _find_label_path(labels_dir: Path, case_id: str, kind: str) -> Path:
    base = labels_dir / case_id / f"{case_id}_{kind}"
    for ext in (".nii.gz", ".nii"):
        candidate = base.with_name(base.name + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {kind} label found for case {case_id} in {labels_dir / case_id} "
        f"(tried {base.name}.nii.gz and {base.name}.nii)"
    )


def _resample_to_reference(image, reference, is_label):
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),  # identity -- images already share the same physical space
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        0,
        image.GetPixelID(),
    )


def load_aligned_case(images_dir: Path, labels_dir: Path, case_id: str):
    """
    Loads all 6 channels for one case and resamples everything onto the
    t2w grid. Returns a dict of SimpleITK images, all sharing t2w's
    size/spacing/origin/direction: {"t2w","adc","hbv","gland","zone","lesion"}
    """
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


# ============================ PREPROCESSING ================================


def resample_to_spacing(image: sitk.Image, target_spacing, is_label: bool) -> sitk.Image:
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [
        max(1, int(round(orig_size[i] * orig_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        image.GetOrigin(),
        target_spacing,
        image.GetDirection(),
        0,
        image.GetPixelID(),
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
    """Resizes the (Y, X) plane only -- Z is already exactly target after crop/pad."""
    assert arr.shape[0] == target_zyx[0], "Z should already match after center_crop_or_pad"
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
    mode = "nearest" if is_label else "trilinear"
    kwargs = {} if is_label else {"align_corners": False}
    resized = F.interpolate(t, size=target_zyx, mode=mode, **kwargs)
    return resized.squeeze(0).squeeze(0).numpy()


def per_slice_znorm(volume_zyx: np.ndarray) -> np.ndarray:
    """Per-slice (each Z index independently) zero-mean unit-std normalization."""
    out = np.empty_like(volume_zyx, dtype=np.float32)
    for z in range(volume_zyx.shape[0]):
        sl = volume_zyx[z]
        m, s = sl.mean(), sl.std()
        out[z] = (sl - m) / s if s > 1e-6 else sl - m
    return out


def preprocess_case(case_dir_images: Path, case_dir_labels: Path, case_id: str):
    """
    Loads one case's 6 native-resolution volumes, resamples everything onto
    a common (t2w) grid (see load_aligned_case above for why zone/adc/hbv
    need this), THEN resamples to the Appendix D target spacing, center crops/pads,
    resizes, and per-slice z-norms the intensity channels.
    Returns (input_5ch, lesion_target) as float32 numpy arrays, shape
    (C, Z, Y, X) and (Z, Y, X) respectively.
    """
    aligned = load_aligned_case(case_dir_images.parent, case_dir_labels.parent, case_id)

    resampled = {
        name: resample_to_spacing(im, TARGET_SPACING, is_label=name in ("gland", "zone", "lesion"))
        for name, im in aligned.items()
    }
    arrs = {name: sitk.GetArrayFromImage(im) for name, im in resampled.items()}  # (Z,Y,X)

    cropped = {name: center_crop_or_pad(arr, CROP_SHAPE_ZYX) for name, arr in arrs.items()}
    resized = {
        name: resize_zyx(arr, FINAL_SHAPE_ZYX, is_label=name in ("gland", "zone", "lesion"))
        for name, arr in cropped.items()
    }

    t2w = per_slice_znorm(resized["t2w"].astype(np.float32))
    adc = per_slice_znorm(resized["adc"].astype(np.float32))
    hbv = per_slice_znorm(resized["hbv"].astype(np.float32))
    gland = (resized["gland"] > 0.5).astype(np.float32)
    zone = resized["zone"].astype(np.float32)  # 0/1/2 -- left as-is, not z-normed (see docstring)
    lesion = (resized["lesion"] > 0.5).astype(np.float32)

    input_5ch = np.stack([t2w, adc, hbv, gland, zone], axis=0)  # (5, Z, Y, X)
    return input_5ch.astype(np.float32), lesion.astype(np.float32)


# ============================== AUGMENTATION ================================


def augment(input_5ch: np.ndarray, target: np.ndarray):
    """Random rotation (+/-15 deg in 5 deg steps) and random flips, applied
    identically to every input channel and the target, in the (Y, X) plane."""
    angle = random.choice(ROTATION_CHOICES_DEG)
    flip_y = random.random() < 0.5
    flip_x = random.random() < 0.5

    def apply(volume_zyx, is_label):
        t = torch.from_numpy(volume_zyx).float().unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
        if angle != 0:
            theta = math.radians(angle)
            cos, sin = math.cos(theta), math.sin(theta)
            affine = torch.tensor([[cos, -sin, 0], [sin, cos, 0]], dtype=torch.float32)
            affine = affine.unsqueeze(0).repeat(t.shape[2], 1, 1)  # per-slice, same angle
            t2d = t.squeeze(0).squeeze(0).unsqueeze(1)  # (Z,1,Y,X)
            grid = F.affine_grid(affine, t2d.shape, align_corners=False)
            mode = "nearest" if is_label else "bilinear"
            t2d = F.grid_sample(t2d, grid, mode=mode, align_corners=False)
            t = t2d.squeeze(1).unsqueeze(0).unsqueeze(0)
        arr = t.squeeze(0).squeeze(0).numpy()
        if flip_y:
            arr = np.flip(arr, axis=1).copy()
        if flip_x:
            arr = np.flip(arr, axis=2).copy()
        return arr

    aug_input = np.stack([apply(input_5ch[c], is_label=(c >= 3)) for c in range(input_5ch.shape[0])], axis=0)
    aug_target = apply(target, is_label=True)
    return aug_input, aug_target


# ============================== DATASET ====================================


class PicaiDataset(Dataset):
    def __init__(self, root: Path, case_ids, augment_data: bool):
        self.root = root
        self.case_ids = case_ids
        self.augment_data = augment_data

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        x, y = preprocess_case(
            self.root / "images" / case_id,
            self.root / "labels" / case_id,
            case_id,
        )
        if self.augment_data:
            x, y = augment(x, y)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_case_split(root: Path):
    import csv
    with open(root / "selected_cases.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    positives = [r["case_id"] for r in rows if r["case_csPCa"] == "YES"]
    negatives = [r["case_id"] for r in rows if r["case_csPCa"] == "NO"]
    random.Random(SEED).shuffle(positives)
    random.Random(SEED).shuffle(negatives)
    n_val_pos = max(1, len(positives) // 5)
    n_val_neg = max(1, len(negatives) // 5)
    val_ids = positives[:n_val_pos] + negatives[:n_val_neg]
    train_ids = positives[n_val_pos:] + negatives[n_val_neg:]
    return train_ids, val_ids


# ================================ MODEL =====================================


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return checkpoint(self.net, x, use_reentrant=False)


class UNet3D(nn.Module):
    """Baseline 3D U-Net. Depth/width kept modest by design -- see flag #1
    on why 1024x1024 spatial resolution forces conservative channel counts
    even with gradient checkpointing + AMP on a 16GB GPU."""

    def __init__(self, in_channels=N_INPUT_CHANNELS, base_ch=16):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.enc1 = ConvBlock(in_channels, chs[0])
        self.enc2 = ConvBlock(chs[0], chs[1])
        self.enc3 = ConvBlock(chs[1], chs[2])
        self.bottleneck = ConvBlock(chs[2], chs[3])
        self.pool = nn.MaxPool3d((1, 2, 2))  # don't downsample the thin Z axis (24 slices)

        self.up3 = nn.ConvTranspose3d(chs[3], chs[2], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = ConvBlock(chs[2] * 2, chs[2])
        self.up2 = nn.ConvTranspose3d(chs[2], chs[1], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = ConvBlock(chs[1] * 2, chs[1])
        self.up1 = nn.ConvTranspose3d(chs[1], chs[0], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlock(chs[0] * 2, chs[0])
        self.out_conv = nn.Conv3d(chs[0], 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)  # logits, (B, 1, Z, Y, X)


# ============================ LOSS / METRIC =================================


class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def dice_on_lesion_positive_slices(pred_logits, target):
    """Dice computed only on slices where the ground-truth lesion mask is
    non-empty, per the paper's own evaluation definition. pred_logits,
    target: (B, 1, Z, Y, X)."""
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
            dice = (2 * intersection / denom).item() if denom > 0 else 1.0
            dices.append(dice)
    return float(np.mean(dices)) if dices else None


# ============================ CHECKPOINTING =================================
#
# Checkpoint filenames encode global_step first so sorting/finding "latest"
# is unambiguous even when a mid-epoch save and an epoch-end save share the
# same epoch number: step_{global_step:07d}_epoch_{epoch:04d}_{tag}.pt

CHECKPOINT_NAME_RE = "step_*_epoch_*_*.pt"


def _global_step_from_name(path: Path) -> int:
    # step_0000123_epoch_0004_mid.pt -> 123
    return int(path.stem.split("_")[1])


def find_latest_checkpoint():
    candidates = []
    if CHECKPOINT_DIR.exists():
        candidates += list(CHECKPOINT_DIR.glob(CHECKPOINT_NAME_RE))
    search_roots = []
    if EXPLICIT_CHECKPOINT_INPUT_DIR is not None:
        search_roots.append(EXPLICIT_CHECKPOINT_INPUT_DIR)
    elif KAGGLE_INPUT_ROOT.exists():
        # auto-discover checkpoints from a previous run re-uploaded as a
        # Kaggle input dataset, without needing to hardcode its slug
        search_roots.append(KAGGLE_INPUT_ROOT)
    for root in search_roots:
        if root.exists():
            candidates += list(root.rglob(CHECKPOINT_NAME_RE))
    if not candidates:
        return None
    return max(candidates, key=_global_step_from_name)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, mid_epoch, history):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "mid" if mid_epoch else "end"
    path = CHECKPOINT_DIR / f"step_{global_step:07d}_epoch_{epoch:04d}_{tag}.pt"
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "mid_epoch": mid_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
    }, path)
    print(f"  Saved checkpoint: {path}")

    if CHECKPOINT_KEEP_LAST_N is not None:
        existing = sorted(CHECKPOINT_DIR.glob(CHECKPOINT_NAME_RE), key=_global_step_from_name)
        for stale in existing[:-CHECKPOINT_KEEP_LAST_N]:
            stale.unlink()


# ====================== POST-TRAINING VALIDATION ============================
#
# Runs automatically at the end of main(), ONLY when SIMULATE_INTERRUPT_
# AFTER_STEPS is None: the interrupt-test path raises SystemExit(0) from
# inside the training loop above, which unwinds out of main() entirely --
# so this code is never reached in interrupt-test mode. No separate flag
# check needed; it falls out of the existing control flow.
#
# Merged in from what used to be two separate notebook cells
# (kaggle_cell2_smoke_test.py and kaggle_cell3_visualize.py) so a single
# committed run produces training + smoke-test + visualization output in
# one continuous log, with no extra cells to paste.


def run_smoke_test_checks(history, checkpoint_dir):
    """Checklist points 1-3 (point 4 is run_visualization() right after
    this). Formerly kaggle_cell2_smoke_test.py's logic."""
    print("\n=== Smoke-test checklist (points 1-3; point 4 follows below) ===\n")

    ok1 = len(history) > 0
    print(f"[1] Training completed epochs without crashing: "
          f"{'PASS' if ok1 else 'FAIL'} ({len(history)} epoch(s) logged)")
    if not ok1:
        print("    -> No epochs logged at all. Check the log above for an exception "
              "before the first epoch finished (missing package, dataset not found, "
              "OOM on the very first forward pass).")

    files = sorted(checkpoint_dir.glob(CHECKPOINT_NAME_RE)) if checkpoint_dir.exists() else []
    ok2 = bool(files)
    print(f"[2] Checkpoint files present: {'PASS' if ok2 else 'FAIL'} ({len(files)} found)")
    if ok2:
        latest = max(files, key=_global_step_from_name)
        try:
            ckpt = torch.load(latest, map_location="cpu")
            required_keys = {"epoch", "global_step", "mid_epoch", "model_state",
                              "optimizer_state", "scheduler_state", "scaler_state", "history"}
            missing = required_keys - set(ckpt.keys())
            ok2 = not missing
            print(f"    Latest checkpoint {latest.name} loads and has all required keys: "
                  f"{'PASS' if ok2 else 'FAIL'}")
            if missing:
                print(f"    -> Missing keys: {missing}")
        except Exception as e:
            ok2 = False
            print(f"    -> FAILED to load {latest}: {e}")
    else:
        print(f"    -> No checkpoints found in {checkpoint_dir}. If training ran, checkpoints "
              "should exist -- check CHECKPOINT_DIR / /kaggle/working permissions.")

    if len(history) < 4:
        ok3 = None
        print(f"[3] Loss trend: SKIPPED (need >=4 epochs to compare meaningfully; "
              f"only {len(history)} logged -- re-check after a longer run)")
    else:
        losses = [h["train_loss"] for h in history]
        n = max(1, len(losses) // 4)
        early_mean = sum(losses[:n]) / n
        late_mean = sum(losses[-n:]) / n
        ok3 = late_mean < early_mean
        print(f"[3] Loss decreasing: {'PASS' if ok3 else 'FAIL'} "
              f"(early mean={early_mean:.4f}, late mean={late_mean:.4f})")
        if not ok3:
            print("    -> Loss flat or rising suggests gradients aren't flowing usefully: check")
            print("       LR, that FOCAL_ALPHA/FOCAL_GAMMA aren't zeroing the loss, and that")
            print("       both train and val splits actually contain positive lesion voxels.")

    print("\n=== Smoke-test summary ===")
    print(f"  1. Completed without crashing:        {'PASS' if ok1 else 'FAIL'}")
    print(f"  2. Checkpoint structurally valid:      {'PASS' if ok2 else 'FAIL'}")
    print(f"  3. Loss decreasing:                    "
          f"{'PASS' if ok3 else ('FAIL' if ok3 is False else 'SKIPPED')}")
    print("  4. Predictions anatomically plausible:  see visualization below")
    print("\nNote: this is a pipeline validity check, not a final model-quality result -- ")
    print("don't read the Dice numbers above as meaningful until the scaling-math conversation")
    print("has set a real epoch/session budget.")


def _best_slice_index(gt: np.ndarray, pred_prob: np.ndarray) -> int:
    score = gt.sum(axis=(1, 2)) * 10 + (pred_prob > 0.5).sum(axis=(1, 2))
    return int(np.argmax(score)) if score.max() > 0 else gt.shape[0] // 2


def _visualize_case(model, root: Path, case_id: str, device, out_dir: Path):
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

    z = _best_slice_index(y, prob)
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
    return degenerate, out_path


def run_visualization(model, root: Path, device, case_ids):
    """Point 4 of the smoke-test checklist. Formerly kaggle_cell3_visualize.py
    -- reuses the already-trained in-memory `model` directly rather than
    reloading from the checkpoint file (that file's loadability was already
    verified by run_smoke_test_checks() above, so this isn't re-testing
    persistence, just visualizing what's already in memory)."""
    print("\n=== Visualization (point 4: anatomically plausible predictions) ===\n")
    model.eval()
    out_dir = KAGGLE_WORKING / "prediction_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    case_ids = case_ids[:VIZ_N_CASES]
    n_degenerate = 0
    out_paths = []
    for cid in case_ids:
        degenerate, out_path = _visualize_case(model, root, cid, device, out_dir)
        n_degenerate += degenerate
        out_paths.append(out_path)

    print(f"\n{len(case_ids) - n_degenerate}/{len(case_ids)} cases look non-degenerate.")
    if n_degenerate > 0:
        print("WARNING: some predictions are degenerate (blank, or covering >50% of the "
              "volume). Before concluding the model is broken: check how many global_steps "
              "it has actually trained for (an undertrained model often predicts all-background "
              "first, since FOCAL_ALPHA=0.97 heavily weights the rare positive class -- give it "
              "more steps before judging), then check FOCAL_ALPHA/FOCAL_GAMMA and LR.")
    print(f"Overlays: {out_dir}")

    try:
        from IPython.display import Image, display
        for p in out_paths:
            display(Image(str(p)))
    except ImportError:
        print("(IPython not available -- overlays saved to disk above but not displayed "
              "inline; expected when run as a plain script, not a notebook cell.)")


# ================================= MAIN =====================================


def print_environment_diagnostics(device):
    """Kaggle's pre-installed Python/PyTorch/CUDA versions aren't something
    this script can know in advance -- print what's actually running so a
    version mismatch (vs. what you tested locally) is visible immediately
    instead of surfacing as a confusing downstream error."""
    import platform
    print(f"Python: {platform.python_version()}")
    print(f"Torch:  {torch.__version__}   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"GPU(s): {n_gpu}")
        for i in range(n_gpu):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name}  {props.total_memory / 1e9:.1f} GB")
        if n_gpu > 1:
            print(f"NOTE: {n_gpu} GPUs detected (e.g. Kaggle's 'GPU T4 x2') but this "
                  f"script only uses cuda:0 -- single-GPU only for this validation run, "
                  f"by design (see docstring flag #5).")
    print(f"SimpleITK: {sitk.Version_VersionString()}")
    if device.type != "cuda":
        print("WARNING: no GPU detected -- this will be extremely slow at "
              f"{FINAL_SHAPE_ZYX} resolution. On Kaggle, enable GPU in "
              "notebook settings (Settings -> Accelerator -> GPU T4 x1 or P100).")


def log_label_extension_summary(root: Path):
    """One-time count of .nii.gz vs bare .nii label files under labels/, so
    if the Kaggle-decompression fallback in _find_label_path() is silently
    active, that's a visible line at startup -- not a guess later if
    training crashes immediately."""
    labels_dir = root / "labels"
    gz_count = sum(1 for _ in labels_dir.rglob("*.nii.gz")) if labels_dir.exists() else 0
    plain_count = sum(1 for _ in labels_dir.rglob("*.nii")) if labels_dir.exists() else 0
    print(f"Label file extensions: {gz_count} as .nii.gz (as packaged), "
          f"{plain_count} as bare .nii (Kaggle-decompression fallback active if > 0)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:0 only, by design
    print(f"Device: {device}")
    print_environment_diagnostics(device)

    root = find_dataset_root()
    print(f"Dataset root: {root}")
    log_label_extension_summary(root)
    train_ids, val_ids = load_case_split(root)
    print(f"Train cases: {len(train_ids)}  Val cases: {len(val_ids)}")

    train_ds = PicaiDataset(root, train_ids, augment_data=True)
    val_ds = PicaiDataset(root, val_ids, augment_data=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_EXP_GAMMA)
    criterion = FocalLoss()
    scaler = make_grad_scaler(device.type)

    start_epoch = 0
    global_step = 0
    history = []
    ckpt_path = find_latest_checkpoint()
    if ckpt_path is not None:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        global_step = ckpt["global_step"]
        # A mid-epoch checkpoint only captured model/optimizer weights, not
        # which cases the dataloader had already consumed -- rather than
        # serialize DataLoader iterator state (fragile across worker
        # processes), we just redo that epoch's data from the start.
        start_epoch = ckpt["epoch"] if ckpt["mid_epoch"] else ckpt["epoch"] + 1
        history = ckpt["history"]
        print(f"  -> global_step={global_step}, resuming at epoch {start_epoch} "
              f"({'redoing' if ckpt['mid_epoch'] else 'continuing after'} epoch {ckpt['epoch']})")

    def maybe_checkpoint_mid_epoch(epoch):
        nonlocal global_step
        if CHECKPOINT_EVERY_N_STEPS and global_step % CHECKPOINT_EVERY_N_STEPS == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                             mid_epoch=True, history=history)
        if SIMULATE_INTERRUPT_AFTER_STEPS and global_step >= SIMULATE_INTERRUPT_AFTER_STEPS:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                             mid_epoch=True, history=history)
            print(f"SIMULATE_INTERRUPT_AFTER_STEPS={SIMULATE_INTERRUPT_AFTER_STEPS} hit "
                  f"at global_step={global_step} -- exiting now to test resume. "
                  f"Set SIMULATE_INTERRUPT_AFTER_STEPS=None at the top of this file and "
                  f"re-run this SAME cell (no kernel restart) to verify it picks up from here.")
            raise SystemExit(0)

    train_start_time = time.time()
    for epoch in range(start_epoch, N_EPOCHS):
        elapsed_hours = (time.time() - train_start_time) / 3600
        if elapsed_hours > MAX_TRAIN_HOURS:
            print(f"Hit MAX_TRAIN_HOURS={MAX_TRAIN_HOURS} -- stopping cleanly, "
                  f"checkpoint saved. Re-run this cell to resume.")
            break

        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).unsqueeze(1)
            with autocast_ctx(device.type):
                logits = model(x)
                loss = criterion(logits, y) / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                maybe_checkpoint_mid_epoch(epoch)
            epoch_loss += loss.item() * ACCUM_STEPS
        scheduler.step()

        model.eval()
        val_dices = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                with autocast_ctx(device.type):
                    logits = model(x)
                d = dice_on_lesion_positive_slices(logits, y)
                if d is not None:
                    val_dices.append(d)
        mean_val_dice = float(np.mean(val_dices)) if val_dices else float("nan")

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{N_EPOCHS}  train_loss={avg_train_loss:.4f}  "
              f"val_dice(lesion+ slices)={mean_val_dice:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed_hours:.2f}h  "
              f"global_step={global_step}")

        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_dice": mean_val_dice,
                         "global_step": global_step})
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                         mid_epoch=False, history=history)

    history_path = KAGGLE_WORKING / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. History: {history_path}")

    # Reaching this point means SIMULATE_INTERRUPT_AFTER_STEPS did NOT fire --
    # that path raises SystemExit(0) from inside the training loop above,
    # which exits before this line is ever reached. So this is either a full
    # N_EPOCHS completion or a clean MAX_TRAIN_HOURS stop -- either way, a
    # real run, not an interrupt test. Run the automated post-training checks
    # in this same process.
    print("\n" + "=" * 78)
    print("POST-TRAINING VALIDATION (smoke-test checks + visualization)")
    print("=" * 78)
    run_smoke_test_checks(history, CHECKPOINT_DIR)
    viz_case_ids = {"train": train_ids, "val": val_ids, "both": train_ids + val_ids}[VIZ_SPLIT]
    run_visualization(model, root, device, viz_case_ids)


if __name__ == "__main__":
    main()
