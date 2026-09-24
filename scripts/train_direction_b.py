"""
Direction B: contrastive concept-region alignment, replacing the original
paper's KKT-constrained alignment objective. Trains on the real 421-case
picai-strict-201pos-220neg dataset, initialized from the completed
baseline U-Net checkpoint (60/60 epochs, V100, final val_dice=0.4030,
peak val_dice=0.4335 at epoch 56).

============================================================================
IMPORTANT CAVEAT -- READ BEFORE TRUSTING THE ARCHITECTURE ASSUMPTIONS BELOW
============================================================================
This script was NOT built by directly reading train.py (the SLURM-adapted
baseline script) -- that file does not exist on the machine this was
written on, and there is no SSH access to the cluster to fetch it. Instead,
this reuses the UNet3D architecture, checkpoint dict format, and train/val
split logic VERBATIM from kaggle_train_baseline_unet.py, the last version
of the training script whose exact content is known, on the assumption
that train.py is a SLURM-mechanics adaptation of it (job submission,
paths, argparse for cluster use) rather than an architectural redesign.

One piece of real evidence supporting this assumption, not just faith:
the completed baseline checkpoint is named step_0005040_epoch_0059_end.pt.
5040 / 60 epochs = 84 optimizer steps/epoch exactly, which is precisely
floor(337 train cases / ACCUM_STEPS=4) -- the same batch-size=1,
accum-steps=4 configuration used here. If train.py's architecture (base_ch,
input channels, checkpoint keys) diverged from what's reproduced below,
torch.load(...).load_state_dict(...) on the real checkpoint will fail
loudly with a key-mismatch error -- it will NOT silently load wrong
weights. If that happens, the fix is reconciling this script's UNet3D
definition with train.py's actual one, not touching the checkpoint.
============================================================================

DESIGN DECISIONS (each justified, not just asserted):

1. CONCEPT VOCABULARY: five categorical concept dimensions read from each
   case's (corrected) rationale JSON -- zone (PZ/TZ/mixed/none), T2W
   intensity bucket (marked/moderate/mild, same thresholds as the
   rationale decision tree), shape (regular/irregular, same 0.6
   sphericity threshold), predicted PI-RADS (1-5), and EPE (present/
   absent). Each gets its own nn.Embedding sub-table; a case's overall
   concept vector is the mean of its five active concept embeddings.
   Concept-region alignment is only meaningful for lesion-POSITIVE cases
   (no real lesion region exists for negatives to contrast against) --
   the contrastive objective trains on the positive subset only.

2. IMAGE REGION EXTRACTOR: the ENTIRE baseline UNet3D (encoder + decoder)
   is loaded from the checkpoint, not just the encoder in isolation. Two
   reasons: (a) this makes Dice directly comparable to the baseline
   number, since it's the identical forward pass; (b) it lets the
   frozen/finetune choice below apply coherently to the whole network
   rather than leaving a decoder trained against features that later
   silently shift out from under it.

   --encoder-mode frozen (DEFAULT, RECOMMENDED):
     Freezes the entire baseline network. Only the new concept embedding
     tables and the region-feature projection head train, via the
     contrastive loss alone. Reasoning: the baseline's encoder already
     learned real prostate/lesion-discriminative features (val_dice
     0.40-0.43 on real held-out cases) -- freezing preserves that rather
     than risking it being overwritten by an unrelated new objective
     trained on a SMALLER real subset (201 real positive cases, vs 337
     real train cases for the original segmentation training). Cheaper
     to train, and gives a clean first answer to "can concept embeddings
     align to the ALREADY-LEARNED feature space at all" before touching
     the segmentation-trained weights at all. Dice in this mode is a
     regression check (should reproduce baseline exactly, since nothing
     in the segmentation path changed), not a new result.

   --encoder-mode finetune:
     Unfreezes the whole network and jointly optimizes
     focal_loss + CONTRASTIVE_WEIGHT * contrastive_loss, exactly like the
     baseline's focal loss but with the new alignment term added. Decoder
     must stay trainable alongside the encoder here -- an encoder whose
     features shift during fine-tuning, feeding a DECODER frozen at its
     old expectations, would degrade segmentation for reasons unrelated
     to whether concept alignment is working. Dice in this mode is a real,
     new number to compare against baseline's 0.4030/0.4335 -- did adding
     the alignment objective help, hurt, or leave segmentation unchanged.

3. CONTRASTIVE LOSS: symmetric InfoNCE with in-batch negatives. Real GPU
   memory forces batch_size=1 for the underlying 3D volumes (matching the
   baseline), so there is no natural "batch" to draw negatives from for a
   single forward/backward step. This script processes CONTRASTIVE_GROUP
   _SIZE real cases sequentially (same one-volume-at-a-time memory
   footprint as the baseline's own ACCUM_STEPS pattern), collects each
   case's concept embedding and pooled region feature, then computes one
   InfoNCE loss across the whole group before stepping the optimizer --
   giving GROUP_SIZE-1 real in-batch negatives per case, at the same
   memory cost per step as the baseline already uses.

4. LOCALIZATION METRIC: primary metric is mean reciprocal rank (MRR) of
   the single highest-similarity true-positive location among all N =
   Z*Yp*Xp bottleneck locations (project every location into
   concept-embedding space, cosine-similarity each against the case's
   concept vector, rank all N, take 1/rank of the best true-positive
   hit). Also reported: that rank as a percentile of N (lower = better).
   R-precision (precision@K with K = true positive voxel count) is kept
   as a secondary/legacy metric for continuity with earlier runs, but it
   is NOT the primary signal: with N ~= 393k and K typically tens to low
   thousands, E[hits under pure-random ranking] = K^2/N rounds to 0, so
   R-precision reads ~0.0000 even when the underlying ranking is
   genuinely, substantially better than chance -- confirmed empirically
   (diagnose_localization_rank.py) on a completed frozen-mode run, where
   R-precision was exactly 0.0000 across all 60 epochs despite best-hit
   ranks averaging the 35th percentile (vs. 50th under pure chance), and
   an early finetune checkpoint showed ranks ~100-1000x better than
   frozen (avg 0.7th percentile) that R-precision still couldn't tell
   apart from frozen's. MRR has no such threshold blind spot.

CHECKPOINTING / RESUME: same pattern as the baseline (global_step-named
checkpoints, mid-epoch saves, rolling retention, resume-by-latest-
checkpoint) -- proven correct there (interrupt/resume tested twice,
commit-to-commit resume tested via simulation) and reused verbatim here.

Usage (cluster-native argparse, NOT a Kaggle single-cell script -- no pip
bootstrap, no /kaggle/input scanning; assumes a conda env with torch +
SimpleITK already working, matching the pre-flight checklist):
  python train_direction_b.py \\
      --data-root ~/picai_data \\
      --rationales-dir ~/picai_data/rationales \\
      --baseline-checkpoint ~/picai_checkpoints/step_0005040_epoch_0059_end.pt \\
      --output-dir ~/picai_direction_b_outputs \\
      --encoder-mode frozen \\
      --n-epochs 30
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset


def make_grad_scaler(device_type: str):
    """Lower init_scale than torch's default (2**16) -- the default is tuned
    for typical classification losses; the InfoNCE contrastive loss here
    divides by temperature=0.07 before softmax, which amplifies gradient
    magnitude ~14x and reliably overflows fp16 on the very first AMP step
    of a freshly-initialized network at the default scale (confirmed via
    local repro: GradScaler silently skips optimizer.step() on overflow,
    which is invisible unless you check get_scale() before/after -- see
    scaler_step_applied below). 2**12 still lets GradScaler grow the scale
    back up automatically once gradients stabilize."""
    try:
        return torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"), init_scale=2.0 ** 12)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=(device_type == "cuda"), init_scale=2.0 ** 12)


def scaler_step_applied(scaler, optimizer) -> bool:
    """Steps optimizer via the GradScaler and returns whether the underlying
    optimizer.step() actually ran. GradScaler.step() calls scaler.update()'s
    counterpart silently: if it detects inf/nan gradients it skips
    optimizer.step() entirely and halves the scale instead of raising --
    scaler.get_scale() is the only way to observe this from the outside."""
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= scale_before


def autocast_ctx(device_type: str):
    try:
        return torch.amp.autocast(device_type, enabled=(device_type == "cuda"))
    except TypeError:
        return torch.cuda.amp.autocast(enabled=(device_type == "cuda"))


# ======================= REUSED FROM THE BASELINE ===========================
# Verbatim (or near-verbatim) from kaggle_train_baseline_unet.py -- see the
# module docstring above for why exact fidelity here matters: any deviation
# would silently mismatch what the pretrained encoder was actually trained
# to see, or fail checkpoint loading (loudly, via a state_dict key error).

TARGET_SPACING = (0.5, 0.5, 3.0)
CROP_SHAPE_ZYX = (24, 384, 384)
FINAL_SHAPE_ZYX = (24, 1024, 1024)
N_INPUT_CHANNELS = 5  # t2w, adc, hbv, gland, zone
SEED = 0  # MUST match the baseline's split seed for Dice to be comparable


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
    """Returns (input_5ch, lesion_target) exactly as the baseline did --
    same target spacing/crop/resize/znorm, since the pretrained encoder's
    features are only meaningful for input preprocessed identically."""
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
    """Identical split logic (same SEED) to the baseline, so Direction B's
    val set is the SAME cases baseline's val_dice=0.4030/0.4335 was
    measured on -- required for the Dice numbers to be comparable at all."""
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
    """Identical to the baseline's UNet3D -- required for the checkpoint's
    state_dict to load. See module docstring for the train.py caveat."""

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
        self.bottleneck_channels = chs[3]  # exposed for the projection head's input size

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.out_conv(d1)
        return logits, b  # (B,1,Z,Y,X) segmentation logits, (B,C,Z,Y',X') bottleneck features


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.97, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


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


# ============================ NEW: DIRECTION B ===============================

ZONE_VOCAB = {"PZ": 0, "TZ": 1, "mixed": 2, "none": 3}
INTENSITY_VOCAB = {"marked": 0, "moderate": 1, "mild": 2}
SHAPE_VOCAB = {"regular": 0, "irregular": 1}
PIRADS_VOCAB = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
EPE_VOCAB = {False: 0, True: 1}

# Same thresholds as the (fixed) rationale decision tree -- bucketing must
# match what the rationale text itself describes, or the concept vocabulary
# would silently disagree with the rationale sentences it's supposed to represent.
MARKED_HYPOINTENSE_Z = -1.2
MODERATE_HYPOINTENSE_Z = -0.6
IRREGULAR_SPHERICITY = 0.6


def rationale_to_concept_indices(rationale: dict):
    zone_idx = ZONE_VOCAB.get(rationale["zone_location"], ZONE_VOCAB["none"])
    z = rationale["t2w_intensity_zscore"]
    if z is None:
        intensity_idx = INTENSITY_VOCAB["mild"]
    elif z <= MARKED_HYPOINTENSE_Z:
        intensity_idx = INTENSITY_VOCAB["marked"]
    elif z <= MODERATE_HYPOINTENSE_Z:
        intensity_idx = INTENSITY_VOCAB["moderate"]
    else:
        intensity_idx = INTENSITY_VOCAB["mild"]
    sph = rationale["shape_sphericity"]
    shape_idx = SHAPE_VOCAB["irregular"] if (sph is not None and sph < IRREGULAR_SPHERICITY) else SHAPE_VOCAB["regular"]
    pirads_idx = PIRADS_VOCAB.get(rationale["predicted_pirads"], PIRADS_VOCAB[2])
    epe_idx = EPE_VOCAB[bool(rationale["extraprostatic_extension"])]
    return zone_idx, intensity_idx, shape_idx, pirads_idx, epe_idx


class ConceptEncoder(nn.Module):
    """Five learned nn.Embedding sub-tables (zone, intensity bucket, shape,
    PI-RADS, EPE) -- no LLM, no pretrained text encoder anywhere. A case's
    concept vector is the mean of its five active embeddings."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.zone_emb = nn.Embedding(len(ZONE_VOCAB), embed_dim)
        self.intensity_emb = nn.Embedding(len(INTENSITY_VOCAB), embed_dim)
        self.shape_emb = nn.Embedding(len(SHAPE_VOCAB), embed_dim)
        self.pirads_emb = nn.Embedding(len(PIRADS_VOCAB), embed_dim)
        self.epe_emb = nn.Embedding(len(EPE_VOCAB), embed_dim)

    def forward(self, zone_idx, intensity_idx, shape_idx, pirads_idx, epe_idx):
        parts = [
            self.zone_emb(zone_idx), self.intensity_emb(intensity_idx), self.shape_emb(shape_idx),
            self.pirads_emb(pirads_idx), self.epe_emb(epe_idx),
        ]
        return torch.stack(parts, dim=0).mean(dim=0)


def masked_average_pool(feature_map: torch.Tensor, lesion_mask_full_res: torch.Tensor):
    """feature_map: (C, Z, Y', X') bottleneck features for one case.
    lesion_mask_full_res: (Z, Y, X) binary mask at full FINAL_SHAPE_ZYX
    resolution. Downsamples the mask to (Z, Y', X') via nearest-neighbor
    (Z is untouched -- the encoder's pool never touches the Z axis) and
    returns the masked-average feature vector, shape (C,), or None if the
    downsampled mask has zero positive voxels."""
    C, Z, Yp, Xp = feature_map.shape
    mask = lesion_mask_full_res.unsqueeze(0).unsqueeze(0).float()
    mask_ds = F.interpolate(mask, size=(Z, Yp, Xp), mode="nearest").squeeze(0).squeeze(0)
    mask_ds = (mask_ds > 0.5).float()
    denom = mask_ds.sum()
    if denom < 1:
        return None
    return (feature_map * mask_ds.unsqueeze(0)).sum(dim=(1, 2, 3)) / denom


def concept_region_contrastive_loss(concept_embeddings: torch.Tensor, region_features: torch.Tensor, temperature: float):
    """Symmetric InfoNCE with in-batch negatives. concept_embeddings,
    region_features: (N, D), N real cases in this group, index-aligned
    (concept_embeddings[i] and region_features[i] are the SAME case).
    Needs N >= 2 for there to be any real negative to push away from."""
    assert concept_embeddings.shape[0] >= 2, "contrastive loss needs >=2 cases per group for real negatives"
    c = F.normalize(concept_embeddings, dim=-1)
    r = F.normalize(region_features, dim=-1)
    logits = c @ r.t() / temperature
    labels = torch.arange(c.shape[0], device=c.device)
    loss_c2r = F.cross_entropy(logits, labels)
    loss_r2c = F.cross_entropy(logits.t(), labels)
    return (loss_c2r + loss_r2c) / 2


def concept_region_localization_metrics(concept_embedding: torch.Tensor, feature_map: torch.Tensor,
                                         region_proj: nn.Module, lesion_mask_full_res: torch.Tensor):
    """Primary metric: reciprocal rank (1/best_rank) of the single
    highest-similarity true-positive location among all N = Z*Yp*Xp
    candidate bottleneck locations (best_rank=1 is perfect, 1-indexed).
    Also returns best_percentile = 100*best_rank/N for reporting, and
    r_precision (precision@K with K = true positive voxel count) as a
    legacy/secondary metric.

    R-precision alone is nearly powerless here: N is ~393k while K is
    typically tens to low thousands, so E[hits under random ranking] =
    K^2/N rounds to 0 even when the underlying ranking is genuinely far
    better than chance -- confirmed empirically (diagnose_localization_
    rank.py) on the frozen-mode run, where R-precision read exactly
    0.0000 across all 60 epochs despite best-voxel ranks sitting at the
    ~35th percentile on average (vs. the 50th percentile expected under
    pure chance), and finetune showed ranks ~100-1000x better than
    frozen that R-precision still couldn't distinguish from frozen's.
    Reciprocal rank has no such threshold effect.

    Returns None if the downsampled mask is empty (lesion vanished at
    bottleneck resolution)."""
    C, Z, Yp, Xp = feature_map.shape
    N = Z * Yp * Xp
    mask = lesion_mask_full_res.unsqueeze(0).unsqueeze(0).float()
    mask_ds = F.interpolate(mask, size=(Z, Yp, Xp), mode="nearest").squeeze(0).squeeze(0) > 0.5
    k = int(mask_ds.sum().item())
    if k == 0:
        return None

    flat_feat = feature_map.permute(1, 2, 3, 0).reshape(-1, C)  # (N, C)
    proj = F.normalize(region_proj(flat_feat), dim=-1)
    concept = F.normalize(concept_embedding, dim=-1)
    sims = proj @ concept  # (N,)
    mask_flat = mask_ds.reshape(-1)

    order = torch.argsort(sims, descending=True)
    true_positive_positions = mask_flat[order].nonzero(as_tuple=True)[0]
    best_rank = int(true_positive_positions.min().item()) + 1  # 1-indexed, 1 = perfect

    hits = mask_flat[order[:k]].sum().item()
    return {
        "reciprocal_rank": 1.0 / best_rank,
        "best_rank": best_rank,
        "best_percentile": 100.0 * best_rank / N,
        "r_precision": hits / k,
    }


class DirectionBDataset(Dataset):
    """positive_only=True restricts to lesion-positive cases (used for the
    contrastive/localization objective and its grouping); False includes
    negatives too (used for the focal-loss segmentation pass in finetune
    mode, matching how the baseline trained on both classes)."""

    def __init__(self, images_dir: Path, labels_dir: Path, rationales_dir: Path, case_ids, positive_only: bool):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.rationales_dir = rationales_dir
        self.case_ids = case_ids
        if positive_only:
            self.case_ids = [c for c in case_ids if (rationales_dir / f"{c}.json").exists()
                              and json.loads((rationales_dir / f"{c}.json").read_text())["lesion_present"]]

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        x, y = preprocess_case(self.images_dir / case_id, self.labels_dir / case_id, case_id)
        rationale_path = self.rationales_dir / f"{case_id}.json"
        rationale = json.loads(rationale_path.read_text()) if rationale_path.exists() else None
        return torch.from_numpy(x), torch.from_numpy(y), case_id, rationale


def direction_b_collate(batch):
    """PyTorch's default collate recursively collates dict VALUES across the
    batch dimension -- rationale["zone_location"] would silently become
    ["PZ"] (a length-1 list) instead of the string "PZ", breaking every
    dict lookup downstream (found by actually running this against real
    data, not assumed). case_id and rationale are kept as plain per-sample
    lists here instead; unwrap with case_ids[0] / rationales[0] at
    batch_size=1 call sites."""
    xs = torch.stack([item[0] for item in batch], dim=0)
    ys = torch.stack([item[1] for item in batch], dim=0)
    case_ids = [item[2] for item in batch]
    rationales = [item[3] for item in batch]
    return xs, ys, case_ids, rationales


# ============================ CHECKPOINTING =================================

CHECKPOINT_NAME_RE = "step_*_epoch_*_*.pt"


def _global_step_from_name(path: Path) -> int:
    return int(path.stem.split("_")[1])


def find_latest_checkpoint(checkpoint_dir: Path):
    candidates = list(checkpoint_dir.glob(CHECKPOINT_NAME_RE)) if checkpoint_dir.exists() else []
    if not candidates:
        return None
    return max(candidates, key=_global_step_from_name)


def save_checkpoint(checkpoint_dir, unet, concept_encoder, region_proj, optimizer, scheduler, scaler,
                     epoch, global_step, mid_epoch, history, encoder_mode, keep_last_n):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = "mid" if mid_epoch else "end"
    path = checkpoint_dir / f"step_{global_step:07d}_epoch_{epoch:04d}_{tag}.pt"
    torch.save({
        "epoch": epoch, "global_step": global_step, "mid_epoch": mid_epoch,
        "encoder_mode": encoder_mode,
        "unet_state": unet.state_dict(),
        "concept_encoder_state": concept_encoder.state_dict(),
        "region_proj_state": region_proj.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
    }, path)
    print(f"  Saved checkpoint: {path}")
    if keep_last_n is not None:
        existing = sorted(checkpoint_dir.glob(CHECKPOINT_NAME_RE), key=_global_step_from_name)
        for stale in existing[:-keep_last_n]:
            stale.unlink()


# ================================= MAIN =====================================


def build_argparser():
    p = argparse.ArgumentParser(description="Direction B: contrastive concept-region alignment")
    p.add_argument("--data-root", type=Path, required=True,
                   help="Folder containing images/, labels/, selected_cases.csv (e.g. ~/picai_data)")
    p.add_argument("--rationales-dir", type=Path, required=True,
                   help="Folder of corrected per-case rationale JSONs from kaggle_rationale_synthesis.py")
    p.add_argument("--baseline-checkpoint", type=Path, required=True,
                   help="Path to the completed baseline checkpoint (step_0005040_epoch_0059_end.pt)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write checkpoints/ (subfolder), training_history.json -- "
                        "same --output-dir convention as the real train.py, so muscle memory "
                        "and any wrapper scripts carry over directly.")
    p.add_argument("--encoder-mode", choices=["frozen", "finetune"], default="frozen",
                   help="See module docstring for the reasoning behind the default (frozen).")
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--lr-exp-gamma", type=float, default=0.97)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--contrastive-weight", type=float, default=0.2,
                   help="Scales the contrastive loss before backward() when --encoder-mode "
                        "finetune, so its gradient onto the SHARED encoder doesn't dominate "
                        "focal_loss's (frozen mode always trains on the contrastive loss at "
                        "full strength -- there's no shared-encoder risk to weigh against, "
                        "see docstring). Confirmed necessary, not theoretical: an unweighted "
                        "(1.0) contrastive pass caused a real finetune run's segmentation to "
                        "collapse to the trivial all-background solution by epoch 6 (val_dice "
                        "0.4030 baseline -> 0.1337 at epoch 0 -> 0.0116 at epoch 5 -> exactly "
                        "0.0 for all 54 remaining epochs), while focal_loss stayed small and "
                        "kept decreasing throughout -- misleadingly, since focal_loss's mean "
                        "reduction over the full ~25M-voxel volume dilutes even a 100%% miss "
                        "on the ~0.006%% of voxels that are lesion-positive into a tiny "
                        "number, so it can't be trusted alone as a health signal once "
                        "collapse starts (see FocalLoss/dice_on_lesion_positive_slices).")
    p.add_argument("--contrastive-warmup-epochs", type=int, default=5,
                   help="In --encoder-mode finetune, the encoder's requires_grad is held "
                        "False for the contrastive pass specifically during these first N "
                        "epochs (concept_encoder/region_proj still train normally against the "
                        "frozen encoder) -- avoids the randomly-initialized concept head "
                        "injecting large, uncalibrated gradients into the pretrained encoder "
                        "before it has learned anything sensible, which is what let the "
                        "collapse above start at epoch 0. The focal/segmentation pass always "
                        "trains the whole network from epoch 0 regardless of this setting -- "
                        "warmup only delays when the CONTRASTIVE pass is allowed to touch the "
                        "shared encoder. Ignored in --encoder-mode frozen (encoder is already "
                        "permanently frozen there).")
    p.add_argument("--contrastive-group-size", type=int, default=4,
                   help="How many real cases' pooled features/embeddings to collect before "
                        "computing one InfoNCE loss and stepping the optimizer -- the "
                        "in-batch-negative analogue of the baseline's ACCUM_STEPS. Larger "
                        "= more negatives per step = more GPU memory (V100 32GB should "
                        "comfortably handle more than the Kaggle T4/P100 16GB could).")
    p.add_argument("--checkpoint-every-n-steps", type=int, default=20)
    p.add_argument("--checkpoint-keep-last-n", type=int, default=3)
    p.add_argument("--max-hours", type=float, default=47.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--viz-n-cases", type=int, default=6,
                   help="Number of val cases to report localization accuracy on per epoch")
    return p


def main():
    args = build_argparser().parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  {props.total_memory / 1e9:.1f} GB")

    checkpoint_dir = args.output_dir / "checkpoints"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = args.data_root / "images"
    labels_dir = args.data_root / "labels"
    train_ids, val_ids = load_case_split(args.data_root)
    print(f"Train cases: {len(train_ids)}  Val cases: {len(val_ids)}")

    train_ds_all = DirectionBDataset(images_dir, labels_dir, args.rationales_dir, train_ids, positive_only=False)
    train_ds_pos = DirectionBDataset(images_dir, labels_dir, args.rationales_dir, train_ids, positive_only=True)
    val_ds_all = DirectionBDataset(images_dir, labels_dir, args.rationales_dir, val_ids, positive_only=False)
    val_ds_pos = DirectionBDataset(images_dir, labels_dir, args.rationales_dir, val_ids, positive_only=True)
    print(f"Train positive-with-rationale cases: {len(train_ds_pos)}  "
          f"Val positive-with-rationale cases: {len(val_ds_pos)}")

    unet = UNet3D().to(device)
    ckpt = torch.load(args.baseline_checkpoint, map_location=device)
    unet.load_state_dict(ckpt["model_state"])  # loud failure here if train.py's architecture diverged
    print(f"Loaded baseline checkpoint: epoch={ckpt['epoch']} global_step={ckpt['global_step']}")

    concept_encoder = ConceptEncoder(args.embed_dim).to(device)
    region_proj = nn.Linear(unet.bottleneck_channels, args.embed_dim).to(device)

    if args.encoder_mode == "frozen":
        for p in unet.parameters():
            p.requires_grad = False
        unet.eval()
        trainable_params = list(concept_encoder.parameters()) + list(region_proj.parameters())
    else:
        trainable_params = list(unet.parameters()) + list(concept_encoder.parameters()) + list(region_proj.parameters())

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_exp_gamma)
    focal_loss_fn = FocalLoss()
    scaler = make_grad_scaler(device.type)

    global_step = 0
    start_epoch = 0
    history = []
    resume_ckpt_path = find_latest_checkpoint(checkpoint_dir)
    if resume_ckpt_path is not None:
        print(f"Resuming from {resume_ckpt_path}")
        rckpt = torch.load(resume_ckpt_path, map_location=device)
        assert rckpt["encoder_mode"] == args.encoder_mode, (
            f"Checkpoint was trained with encoder_mode={rckpt['encoder_mode']}, "
            f"but --encoder-mode={args.encoder_mode} was requested -- these are not resumable "
            f"into each other (different trainable-parameter sets)."
        )
        unet.load_state_dict(rckpt["unet_state"])
        concept_encoder.load_state_dict(rckpt["concept_encoder_state"])
        region_proj.load_state_dict(rckpt["region_proj_state"])
        optimizer.load_state_dict(rckpt["optimizer_state"])
        scheduler.load_state_dict(rckpt["scheduler_state"])
        scaler.load_state_dict(rckpt["scaler_state"])
        global_step = rckpt["global_step"]
        start_epoch = rckpt["epoch"] if rckpt["mid_epoch"] else rckpt["epoch"] + 1
        history = rckpt["history"]
        print(f"  -> global_step={global_step}, resuming at epoch {start_epoch}")

    train_loader_pos = DataLoader(train_ds_pos, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                   collate_fn=direction_b_collate)
    train_loader_all = DataLoader(train_ds_all, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                   collate_fn=direction_b_collate)

    def compute_case_embeddings(x, y, rationale):
        """Runs one case through the (possibly frozen) encoder, returns
        (concept_embedding, region_feature, bottleneck_feature_map) or
        (None, None, None) if the case has no usable lesion region."""
        x = x.to(device)
        with autocast_ctx(device.type):
            _, bottleneck = unet(x)
        bottleneck = bottleneck[0].float()  # drop batch dim (batch_size=1), back to fp32 for the loss math
        lesion_mask = y[0].to(device)
        region_feat = masked_average_pool(bottleneck, lesion_mask)
        if region_feat is None:
            return None, None, None
        region_feat = region_proj(region_feat)
        zone_idx, int_idx, shape_idx, pirads_idx, epe_idx = rationale_to_concept_indices(rationale)
        idx_t = lambda v: torch.tensor(v, device=device)
        concept_emb = concept_encoder(idx_t(zone_idx), idx_t(int_idx), idx_t(shape_idx), idx_t(pirads_idx), idx_t(epe_idx))
        return concept_emb, region_feat, bottleneck

    train_start_time = time.time()
    for epoch in range(start_epoch, args.n_epochs):
        elapsed_hours = (time.time() - train_start_time) / 3600
        if elapsed_hours > args.max_hours:
            print(f"Hit --max-hours={args.max_hours} -- stopping cleanly, checkpoint saved.")
            break

        if args.encoder_mode == "finetune":
            unet.train()
        concept_encoder.train()
        region_proj.train()

        # Contrastive-pass warmup (finetune only): hold the shared encoder's
        # requires_grad False for the contrastive pass during the first
        # --contrastive-warmup-epochs epochs, so a randomly-initialized
        # concept_encoder/region_proj can't inject large, uncalibrated
        # gradients into the pretrained encoder before they've learned
        # anything sensible -- confirmed root cause of a real segmentation
        # collapse (see --contrastive-weight help text). Restored to
        # trainable right after, unconditionally, so the focal pass below
        # always trains the whole network regardless of warmup status.
        in_contrastive_warmup = (args.encoder_mode == "finetune"
                                  and epoch < args.contrastive_warmup_epochs)
        if args.encoder_mode == "finetune" and in_contrastive_warmup:
            for p in unet.parameters():
                p.requires_grad = False

        # --- contrastive pass over positive-with-rationale cases, grouped ---
        group_concepts, group_regions = [], []
        epoch_contrastive_loss, n_contrastive_steps = 0.0, 0
        optimizer.zero_grad()
        for x, y, case_ids, rationales in train_loader_pos:
            rationale = rationales[0]  # batch_size=1; direction_b_collate keeps this a real dict
            concept_emb, region_feat, _ = compute_case_embeddings(x, y, rationale)
            if concept_emb is None:
                continue
            group_concepts.append(concept_emb)
            group_regions.append(region_feat)
            if len(group_concepts) >= args.contrastive_group_size:
                c = torch.stack(group_concepts, dim=0)
                r = torch.stack(group_regions, dim=0)
                with autocast_ctx(device.type):
                    loss = concept_region_contrastive_loss(c, r, args.temperature)
                # Scaled down in finetune mode so this pass's gradient onto the
                # SHARED encoder doesn't dominate focal_loss's -- frozen mode
                # has no shared-encoder risk, so it stays at full strength.
                # epoch_contrastive_loss logs the RAW (unweighted) loss value
                # so it stays comparable across --contrastive-weight settings.
                backward_loss = loss * args.contrastive_weight if args.encoder_mode == "finetune" else loss
                scaler.scale(backward_loss).backward()
                applied = scaler_step_applied(scaler, optimizer)
                optimizer.zero_grad()
                group_concepts, group_regions = [], []
                if applied:
                    # only count steps GradScaler actually applied -- otherwise
                    # this group's loss never produced a real weight update and
                    # counting it (a) misrepresents epoch_contrastive_loss and
                    # (b) leaves optimizer.step() looking "never called" to the
                    # LR scheduler if it's the only group this epoch, which is
                    # exactly what produced the "lr_scheduler.step() called
                    # before optimizer.step()" warning on the V100 run.
                    epoch_contrastive_loss += loss.item()
                    n_contrastive_steps += 1
                    global_step += 1
                    if args.checkpoint_every_n_steps and global_step % args.checkpoint_every_n_steps == 0:
                        save_checkpoint(checkpoint_dir, unet, concept_encoder, region_proj, optimizer,
                                         scheduler, scaler, epoch, global_step, True, history, args.encoder_mode,
                                         args.checkpoint_keep_last_n)
                else:
                    print(f"  [warn] contrastive step skipped by GradScaler (grad overflow) "
                          f"at global_step={global_step}, new scale={scaler.get_scale():.0f}")

        # --- segmentation pass (finetune mode only) over all cases ---
        epoch_focal_loss, n_focal_steps = 0.0, 0
        if args.encoder_mode == "finetune":
            if in_contrastive_warmup:
                for p in unet.parameters():
                    p.requires_grad = True
            for x, y, case_ids, rationales in train_loader_all:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                with autocast_ctx(device.type):
                    logits, _ = unet(x)
                    loss = focal_loss_fn(logits, y)
                scaler.scale(loss).backward()
                applied = scaler_step_applied(scaler, optimizer)
                optimizer.zero_grad()
                if applied:
                    epoch_focal_loss += loss.item()
                    n_focal_steps += 1
                    global_step += 1
                else:
                    print(f"  [warn] focal step skipped by GradScaler (grad overflow) "
                          f"at global_step={global_step}, new scale={scaler.get_scale():.0f}")

        scheduler.step()

        # --- evaluation: Dice + localization accuracy on val set ---
        unet.eval()
        concept_encoder.eval()
        region_proj.eval()
        val_dices = []
        with torch.no_grad():
            for x, y, case_ids, rationales in DataLoader(val_ds_all, batch_size=1, num_workers=args.num_workers,
                                                          collate_fn=direction_b_collate):
                x, y = x.to(device), y.to(device).unsqueeze(1)
                with autocast_ctx(device.type):
                    logits, _ = unet(x)
                d = dice_on_lesion_positive_slices(logits, y)
                if d is not None:
                    val_dices.append(d)
        mean_val_dice = float(np.mean(val_dices)) if val_dices else None

        loc_mrrs, loc_percentiles, loc_accs = [], [], []
        with torch.no_grad():
            for i, (x, y, case_ids, rationales) in enumerate(DataLoader(val_ds_pos, batch_size=1, num_workers=args.num_workers,
                                                                          collate_fn=direction_b_collate)):
                if i >= args.viz_n_cases:
                    break
                concept_emb, _, bottleneck = compute_case_embeddings(x, y, rationales[0])
                if concept_emb is None:
                    continue
                metrics = concept_region_localization_metrics(concept_emb, bottleneck, region_proj, y[0].to(device))
                if metrics is not None:
                    loc_mrrs.append(metrics["reciprocal_rank"])
                    loc_percentiles.append(metrics["best_percentile"])
                    loc_accs.append(metrics["r_precision"])
        # Primary localization metric -- see concept_region_localization_metrics'
        # docstring for why R-precision (mean_loc_acc, kept below as legacy)
        # is nearly powerless to detect real ranking differences at this N.
        mean_loc_mrr = float(np.mean(loc_mrrs)) if loc_mrrs else None
        mean_loc_percentile = float(np.mean(loc_percentiles)) if loc_percentiles else None
        mean_loc_acc = float(np.mean(loc_accs)) if loc_accs else None

        avg_contrastive = epoch_contrastive_loss / n_contrastive_steps if n_contrastive_steps else None
        # None in frozen mode by design (no finetune weight update to combine
        # focal loss with, see module docstring) -- NOT a NaN/crash. Also None
        # whenever every step in the epoch was skipped by GradScaler (see
        # scaler_step_applied above and the [warn] lines during training).
        avg_focal = epoch_focal_loss / n_focal_steps if n_focal_steps else None

        def fmt(v):
            return f"{v:.4f}" if v is not None else "n/a"

        warmup_tag = " [contrastive_warmup: encoder frozen for contrastive pass]" if in_contrastive_warmup else ""
        print(f"Epoch {epoch+1}/{args.n_epochs}  contrastive_loss={fmt(avg_contrastive)}  "
              f"focal_loss={fmt(avg_focal)}  val_dice(lesion+ slices)={fmt(mean_val_dice)}  "
              f"val_localization_mrr(n={len(loc_mrrs)})={fmt(mean_loc_mrr)}  "
              f"val_localization_percentile={fmt(mean_loc_percentile)}  "
              f"val_localization_accuracy(R-precision, legacy)={fmt(mean_loc_acc)}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed_hours:.2f}h  "
              f"global_step={global_step}{warmup_tag}")

        history.append({"epoch": epoch, "contrastive_loss": avg_contrastive, "focal_loss": avg_focal,
                         "contrastive_warmup": in_contrastive_warmup,
                         "val_dice": mean_val_dice, "val_localization_mrr": mean_loc_mrr,
                         "val_localization_percentile": mean_loc_percentile,
                         "val_localization_accuracy": mean_loc_acc,
                         "global_step": global_step})
        save_checkpoint(checkpoint_dir, unet, concept_encoder, region_proj, optimizer, scheduler, scaler,
                         epoch, global_step, False, history, args.encoder_mode, args.checkpoint_keep_last_n)

    history_path = args.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. History: {history_path}")


if __name__ == "__main__":
    main()
