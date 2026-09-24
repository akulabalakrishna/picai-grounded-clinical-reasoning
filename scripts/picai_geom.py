"""
Shared alignment logic used by BOTH the local sanity-check script
(03_sanity_check.py) and the Kaggle training script. Keeping this in one
module means the sanity-check overlays and the actual training inputs are
guaranteed to be built the same way -- no drift between "what we visually
verified" and "what the model actually sees".

Ground truth about this dataset's geometry (confirmed by inspection, not
assumed):
  - t2w, gland (whole_gland/AI/Bosma22b), and lesion
    (csPCa_lesion_delineations/human_expert/resampled) already share the
    exact same size/spacing/origin/direction. No resampling is a no-op for
    these but we resample anyway for safety -- it's idempotent.
  - zone (zonal_pz_tz/AI/HeviAI23) shares spacing with t2w but is cropped
    to a smaller bounding box (different size/origin) -- MUST be resampled
    onto the t2w grid or it silently misaligns.
  - adc and hbv are natively lower in-plane resolution (larger spacing,
    smaller size) than t2w -- MUST be resampled onto the t2w grid.

All resampling here uses t2w as the reference frame. Label maps (gland,
zone, lesion) use nearest-neighbor interpolation; intensity images (adc,
hbv) use linear interpolation. Default pixel value for out-of-bounds
regions is 0 (background / no signal), which is correct for both label
maps and intensity images in this dataset.
"""
from pathlib import Path

import numpy as np
import SimpleITK as sitk


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
    size/spacing/origin/direction:
        {"t2w", "adc", "hbv", "gland", "zone", "lesion"}
    """
    t2w = sitk.ReadImage(str(images_dir / case_id / f"{case_id}_t2w.mha"))
    adc = sitk.ReadImage(str(images_dir / case_id / f"{case_id}_adc.mha"))
    hbv = sitk.ReadImage(str(images_dir / case_id / f"{case_id}_hbv.mha"))
    gland = sitk.ReadImage(str(labels_dir / case_id / f"{case_id}_gland.nii.gz"))
    zone = sitk.ReadImage(str(labels_dir / case_id / f"{case_id}_zone.nii.gz"))
    lesion = sitk.ReadImage(str(labels_dir / case_id / f"{case_id}_lesion.nii.gz"))

    return {
        "t2w": t2w,
        "adc": _resample_to_reference(adc, t2w, is_label=False),
        "hbv": _resample_to_reference(hbv, t2w, is_label=False),
        "gland": _resample_to_reference(sitk.Cast(gland, sitk.sitkUInt8), t2w, is_label=True),
        "zone": _resample_to_reference(sitk.Cast(zone, sitk.sitkUInt8), t2w, is_label=True),
        "lesion": _resample_to_reference(sitk.Cast(lesion, sitk.sitkUInt8), t2w, is_label=True),
    }


def aligned_case_to_arrays(aligned: dict):
    """Converts an aligned-case dict (SimpleITK images) to numpy arrays, shape (Z, Y, X)."""
    return {k: sitk.GetArrayFromImage(v) for k, v in aligned.items()}


def assert_all_aligned(aligned: dict):
    ref = aligned["t2w"]
    for name, im in aligned.items():
        assert im.GetSize() == ref.GetSize(), f"{name} size {im.GetSize()} != t2w {ref.GetSize()}"
        assert np.allclose(im.GetSpacing(), ref.GetSpacing(), atol=1e-3), f"{name} spacing mismatch"
        assert np.allclose(im.GetOrigin(), ref.GetOrigin(), atol=1e-2), f"{name} origin mismatch"
