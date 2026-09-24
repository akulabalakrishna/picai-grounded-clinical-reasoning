"""
Kaggle-notebook-ready training script: baseline 3D U-Net for csPCa lesion
segmentation, conditioned on anatomical-prior masks, matching the paper's
Appendix D spec as closely as a free Kaggle GPU allows.

SETUP ON KAGGLE
---------------
1. Upload the data as a Kaggle Dataset. Either works -- find_dataset_root()
   below auto-detects both, and does NOT assume a specific dataset slug:
     (a) `kaggle datasets create -p E:\Cancer_IITH\picai\kaggle_upload` --
         uploads the folder directly (recommended: no zip-extraction
         ambiguity), or
     (b) upload picai_sample_421cases.zip via the web UI
         (kaggle.com/datasets/new) -- whether Kaggle auto-extracts it or
         leaves it as a .zip, find_dataset_root() handles both.
2. IMPORTANT -- SimpleITK is NOT in Kaggle's default GPU image. Run this in
   the first notebook cell before anything else (needs "Internet: On" in
   notebook Settings, or this pip install fails with a connection error):
       !pip install -q SimpleITK
3. Create a new Notebook, add that dataset, turn on a GPU (T4 x1, T4 x2, or
   P100 -- whatever's available; see the multi-GPU note in print_environment
   _diagnostics() if you get a dual T4).
4. Copy this file into a notebook cell (or `%run` it / `!python train.py`
   after uploading it as a Kaggle "Utility script").
5. Run. It prints Python/torch/CUDA/SimpleITK versions and GPU count/memory
   at startup -- check these against what you tested locally before trusting
   the run. Checkpoints land in /kaggle/working/checkpoints (the only
   writable path, 20GB quota -- CHECKPOINT_KEEP_LAST_N below keeps this from
   filling up over a long run). "Save Version" before your session ends so
   checkpoints aren't lost, or download them, or re-upload
   /kaggle/working/checkpoints as a new input dataset for the next session
   (auto-discovered by find_latest_checkpoint(), no path editing needed --
   see EXPLICIT_CHECKPOINT_INPUT_DIR only if auto-discovery picks up more
   than one candidate run and you need to disambiguate).
6. To verify checkpoint/resume actually works BEFORE trusting an 8-hour run,
   set SIMULATE_INTERRUPT_AFTER_STEPS = 5 below, run once (it saves a
   checkpoint and exits after 5 optimizer steps), then set it back to None
   and run again -- it should resume from global_step 5, not restart from 0.

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
     16GB T4/P100. A 3D U-Net's activations at the first encoder level alone
     (e.g. 32ch @ 24x1024x1024, fp16) are ~1.6GB EACH, and backprop needs to
     keep many of these around. This script defaults to:
       - batch_size = 1 (BATCH_SIZE below)
       - automatic mixed precision (torch.cuda.amp)
       - gradient checkpointing on every U-Net block (torch.utils.checkpoint)
       - gradient accumulation (ACCUM_STEPS) to simulate a larger batch
     If you still hit CUDA OOM on a T4 (P100 has no tensor cores but more
     straightforward fp32 headroom in practice), set DOWNSCALE_FACTOR=2
     below (trains at 512x512 instead of 1024x1024) -- this is a real
     deviation from the spec, which is why it's a visible flag, not a
     silent default.

  2. SESSION TIME LIMIT: Kaggle free sessions cut off at ~9-12h with no
     warning grace period. This script checkpoints every CHECKPOINT_EVERY_N
     _STEPS optimizer steps (mid-epoch, not just at epoch boundaries) AND
     tracks wall-clock time, stopping cleanly and saving before
     MAX_TRAIN_HOURS is hit (default 8.5h, leaving headroom). On restart, it
     auto-resumes from the latest checkpoint by scanning
     /kaggle/working/checkpoints AND /kaggle/input (so a previous run's
     checkpoints, re-uploaded as a new input dataset, are found without
     editing any path) -- set EXPLICIT_CHECKPOINT_INPUT_DIR only if that
     auto-scan is ambiguous (e.g. multiple old runs attached at once).

  3. DISK: /kaggle/working has a 20GB quota (typical). This sample's
     packaged data is ~4.4GB (421 cases) so this is still a non-issue, but
     will matter once you scale to the full ~1500-case dataset -- you
     will likely need to stream/preprocess-on-the-fly (as this script
     already does) rather than caching a fully-resampled copy to disk.

  4. SAMPLE SIZE: 421 cases (201 real human-expert-verified positive + 220
     negative -- STRICT policy, no AI-derived masks; 337 train / 84 val,
     stratified by csPCa) is the first real baseline run, not a final
     result. Note: of the 220 real-verified positives in picai_labels,
     19 (8.6%) failed mask-alignment verification and were excluded before
     this package was built -- see picai/findings.md. Dice numbers from
     this run confirm the pipeline runs correctly end-to-end on real data
     at scale -- they are not
     meaningful model-quality numbers. Scale up before drawing conclusions.
"""
import json
import math
import random
import time
import zipfile
from pathlib import Path

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
SIMULATE_INTERRUPT_AFTER_STEPS = None  # int for testing: e.g. 5 -- the run saves a
                                        # checkpoint and exits after this many global
                                        # optimizer steps, so you can verify resume
                                        # works in a two-minute test instead of only
                                        # finding out after Kaggle kills an 8-hour run.
                                        # Set back to None for a real training run.

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
# Inlined here (was previously a separate picai_geom.py imported via a
# dataset-root sys.path hack) so this script is fully self-contained -- no
# cross-file dependency on whatever picai_geom.py copy happens to be
# bundled in a given dataset version. Same alignment logic used by
# 03_sanity_check.py locally.
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
#     dataset's files via the API, not assumed) to silently decompress
#     gzip'd files uploaded inside a zip-mode directory upload --
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
                  f"script only uses cuda:0 -- see FLAG in scaling notes for adding "
                  f"nn.DataParallel if you want to use both.")
    print(f"SimpleITK: {sitk.Version_VersionString()}")
    if device.type != "cuda":
        print("WARNING: no GPU detected -- this will be extremely slow at "
              f"{FINAL_SHAPE_ZYX} resolution. On Kaggle, enable GPU in "
              "notebook settings (Settings -> Accelerator -> GPU T4 x1 or P100).")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print_environment_diagnostics(device)

    root = find_dataset_root()
    print(f"Dataset root: {root}")
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
        # processes), we just redo that epoch's data from the start. With a
        # 24-case train set the wasted work is trivial and correctness (no
        # silently-skipped cases, no double-counted epoch) matters more.
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
                  f"Set SIMULATE_INTERRUPT_AFTER_STEPS=None and re-run to verify it "
                  f"picks up from here (see the checkpoint-verification checklist).")
            raise SystemExit(0)

    train_start_time = time.time()
    for epoch in range(start_epoch, N_EPOCHS):
        elapsed_hours = (time.time() - train_start_time) / 3600
        if elapsed_hours > MAX_TRAIN_HOURS:
            print(f"Hit MAX_TRAIN_HOURS={MAX_TRAIN_HOURS} -- stopping cleanly, "
                  f"checkpoint saved. Re-run this cell/script to resume.")
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


if __name__ == "__main__":
    main()
