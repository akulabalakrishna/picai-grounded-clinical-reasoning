"""
SINGLE-CELL, SELF-CONTAINED Kaggle script for Direction A: rule-based
rationale synthesis directly against the real picai-strict-201pos-220neg
dataset. No LLM anywhere -- every rationale is computed from real mask
geometry and real T2W pixel statistics, then filled into a template
sentence via a hand-ported PI-RADS-style decision tree.

CPU only. No GPU requested, no torch import -- this is pure SimpleITK
measurement + arithmetic + string templating.

Paste this whole file into one Kaggle notebook cell (or run as a script
kernel) and run it. Only external dependency needing install is
SimpleITK + scikit-image (for marching-cubes sphericity) -- done as a
subprocess call below, same pattern as the training script.

============================================================================
GROUND TRUTH ABOUT THIS DATASET'S GEOMETRY (confirmed by inspection when
this pipeline was first built, and re-confirmed for the training script on
Kaggle's real filesystem):
  - t2w, gland, and lesion masks share the exact same size/spacing/origin/
    direction as packaged. zone shares spacing but is cropped to a smaller
    bounding box -- MUST be resampled onto the t2w grid. adc/hbv are
    natively lower in-plane resolution -- MUST be resampled too.
  - Label file extension: shipped locally as .nii.gz, but Kaggle's dataset
    ingestion silently decompresses gzip'd files uploaded inside a
    zip-mode directory upload -- labels/<case>/<case>_gland.nii.gz comes
    back as ..._gland.nii on Kaggle. _find_label_path() tries both.

DATA POLICY NOTE: even though selected_cases.csv already reflects a prior
local sanity-check pass (19 of 220 real-verified positives excluded before
packaging), this script RE-RUNS the mask-alignment defect check from
scratch against the real uploaded data -- verifying data integrity in the
same place computation actually happens, not trusting a prior local pass
blindly. Any case failing here is excluded from rationale output and
logged, exactly as the original local pipeline did.

CLINICAL LIMITATION, STATED EXPLICITLY (not silently ignored): PI-CAI is a
BIPARAMETRIC dataset (T2W + ADC + HBV/DWI only, no DCE). Real PI-RADS v2.1
uses DCE strictly as a tie-breaker for peripheral-zone category-3 lesions
(early enhancement can upgrade PZ-3 to PZ-4). This decision tree cannot
apply that tie-breaker and says so explicitly in any PZ category-3
rationale, rather than silently guessing an upgrade it has no evidence for.
============================================================================
"""
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
     "SimpleITK", "scikit-image"],
    check=True,
)

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes, mesh_surface_area

# ============================== CONFIG ====================================

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
RATIONALES_DIR = KAGGLE_WORKING / "rationales"

EXPLICIT_DATASET_ROOT = None  # e.g. Path("/kaggle/input/picai-strict-201pos-220neg")

PZ_LABEL, TZ_LABEL = 1, 2

# Defect-check thresholds (mask-alignment verification, re-run here against
# the real uploaded data rather than trusted from the prior local pass).
LESION_MOSTLY_IN_GLAND_FRACTION = 0.5   # >50% of lesion voxels must sit inside gland
CENTROID_DISTANCE_RADIUS_MULTIPLE = 1.5  # lesion centroid must be within 1.5x the
                                          # gland's equivalent-sphere radius of the
                                          # gland centroid -- an independent geometric
                                          # signal alongside the overlap-fraction check
                                          # (an annotation defect can in principle have
                                          # some voxel overlap by chance while the bulk
                                          # of the lesion sits far from the gland)

# Rationale decision-tree thresholds -- calibrated against a 15-case real
# positive pilot sample earlier in this project (see project history);
# carried over unchanged here, not re-calibrated at full 421-case scale
# since that wasn't requested.
LARGE_LESION_MM3 = 1500.0        # 1.5 cc -- PI-RADS v2.1 size threshold
MARKED_HYPOINTENSE_Z = -1.2
MODERATE_HYPOINTENSE_Z = -0.6
IRREGULAR_SPHERICITY = 0.6        # below this = irregular/ill-defined margin
EPE_FRACTION_THRESHOLD = 0.30     # fraction of lesion outside gland mask to flag
                                   # possible extraprostatic extension in the
                                   # rationale text -- deliberately conservative
                                   # because the gland mask is AI-derived, not
                                   # human-verified at the capsule boundary


# ============================ DATA LOCATION ================================


def find_dataset_root() -> Path:
    """Locates the picai dataset folder under /kaggle/input. Does NOT assume
    a specific dataset slug -- searches for an "images" dir with a sibling
    "labels" dir, at any depth. Same proven logic as the training script."""
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

    local = Path(r"E:\Cancer_IITH\picai\kaggle_upload")  # off-Kaggle local test path
    if (local / "images").exists():
        return local
    raise FileNotFoundError(
        "Could not find the picai dataset under /kaggle/input or locally. "
        "Did you add the Kaggle Dataset to this notebook?"
    )


# ============================ GEOMETRY / ALIGNMENT ==========================


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
        image, reference, sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        0, image.GetPixelID(),
    )


def load_aligned_case(images_dir: Path, labels_dir: Path, case_id: str):
    """Loads all 6 channels for one case and resamples everything onto the
    t2w grid at NATIVE resolution (no crop/resize -- this is measurement,
    not model input prep, so real physical scale is what matters)."""
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


def aligned_case_to_arrays(aligned: dict):
    return {k: sitk.GetArrayFromImage(v) for k, v in aligned.items()}  # (Z,Y,X)


# ============================ DEFECT CHECK ==================================


def voxel_volume_mm3(sitk_image):
    sx, sy, sz = sitk_image.GetSpacing()
    return sx * sy * sz


def _centroid_mm(mask_bool, spacing_zyx):
    idx = np.argwhere(mask_bool)
    if len(idx) == 0:
        return None
    centroid_voxel = idx.mean(axis=0)  # (z, y, x)
    return centroid_voxel * np.array(spacing_zyx)  # mm, same axis order


def run_defect_check(case_id, is_positive, arrs, spacing_zyx, voxel_vol):
    """Re-verifies mask alignment against the REAL uploaded data (not
    trusted from the prior local pass). Returns (passed: bool, details: dict)."""
    gland = arrs["gland"] > 0
    lesion = arrs["lesion"] > 0

    gland_vox = int(gland.sum())
    lesion_vox = int(lesion.sum())
    lesion_in_gland_vox = int((lesion & gland).sum())
    frac_lesion_in_gland = lesion_in_gland_vox / lesion_vox if lesion_vox > 0 else None

    gland_centroid = _centroid_mm(gland, spacing_zyx)
    lesion_centroid = _centroid_mm(lesion, spacing_zyx)
    centroid_distance_mm = None
    gland_equiv_radius_mm = None
    centroid_ok = True
    if gland_centroid is not None:
        gland_volume_mm3 = gland_vox * voxel_vol
        gland_equiv_radius_mm = (3 * gland_volume_mm3 / (4 * np.pi)) ** (1 / 3)
        if lesion_centroid is not None:
            centroid_distance_mm = float(np.linalg.norm(gland_centroid - lesion_centroid))
            centroid_ok = centroid_distance_mm <= CENTROID_DISTANCE_RADIUS_MULTIPLE * gland_equiv_radius_mm

    checks = {
        "gland_nonempty": gland_vox > 0,
        "lesion_matches_label": (lesion_vox > 0) == is_positive,
        "lesion_mostly_in_gland": (
            frac_lesion_in_gland is None or frac_lesion_in_gland > LESION_MOSTLY_IN_GLAND_FRACTION
        ),
        "lesion_centroid_near_gland": centroid_ok,
    }
    passed = all(checks.values())
    details = {
        "gland_voxels": gland_vox,
        "lesion_voxels": lesion_vox,
        "frac_lesion_in_gland": round(frac_lesion_in_gland, 3) if frac_lesion_in_gland is not None else None,
        "centroid_distance_mm": round(centroid_distance_mm, 1) if centroid_distance_mm is not None else None,
        "gland_equiv_radius_mm": round(gland_equiv_radius_mm, 1) if gland_equiv_radius_mm is not None else None,
        "checks": checks,
    }
    return passed, details


# ============================ RATIONALE SYNTHESIS ===========================


def sphericity(mask_arr, spacing_zyx):
    """(36*pi*V^2)^(1/3) / surface_area; 1.0 = sphere, lower = irregular.
    Marching-cubes on the real anisotropic spacing (0.3mm in-plane vs 3mm
    slice thickness) -- a naive voxel-face-counting surface area is badly
    biased by that anisotropy (found and discarded earlier in this project)."""
    volume_mm3 = float(mask_arr.sum() * np.prod(spacing_zyx))
    if volume_mm3 <= 0:
        return None
    padded = np.pad(mask_arr.astype(np.uint8), 1)
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing_zyx)
    except (RuntimeError, ValueError):
        return None  # lesion too small/thin for a closed surface
    area_mm2 = mesh_surface_area(verts, faces)
    if area_mm2 <= 0:
        return None
    return float((36 * np.pi * volume_mm3 ** 2) ** (1 / 3) / area_mm2)


def classify_pirads(zone, size_mm3, z_score, shape_sphericity, epe):
    """Hand-ported, simplified PI-RADS v2.1-style decision tree.

    FIX (2026-09-18): the original version had `if epe or (marked and
    large): return 5` -- EPE ALONE short-circuited straight to the most
    severe category regardless of how mild every other measurement was.
    Found via case 11357_1001380: z=-0.159 (mildly hypointense, the
    weakest bucket), 0.385cc (well under the 1.5cc "large" threshold),
    sphericity=0.639 (round, not irregular) -- every real measurement is
    unremarkable, yet EPE=True alone drove predicted_pirads to 5 against
    a real marksheet score of 2.

    Fixed by computing a BASE category from intensity/size/shape/zone
    alone (identical logic to before, just without the EPE override), and
    only THEN letting EPE act as a one-category upgrade (capped at 5) --
    gated on `marked or large` (matches the real underlying finding being
    at least one of: markedly hypointense, or a large lesion), not on the
    base category already being >=3. Verified against real cases that this
    gate choice matters: a 40cc irregular mass with EPE but only mild mean
    T2 z-score (case 10522_1000532) should still be allowed an EPE-driven
    upgrade because of its size, even though its z-score alone wouldn't
    reach "moderate". EPE reinforces an already-present severe finding
    (marked hypointensity or large size); it cannot manufacture suspicion
    out of a lesion that is BOTH mild-intensity AND small.
    """
    marked = z_score <= MARKED_HYPOINTENSE_Z
    moderate = MODERATE_HYPOINTENSE_Z >= z_score > MARKED_HYPOINTENSE_Z
    irregular = shape_sphericity is not None and shape_sphericity < IRREGULAR_SPHERICITY
    large = size_mm3 >= LARGE_LESION_MM3

    if zone == "PZ":
        if marked and large:
            base = 5
        elif marked and not large:
            base = 4
        elif moderate or (marked and irregular):
            base = 3
        else:
            base = 2
    else:  # TZ or mixed, treated conservatively like TZ
        if marked and large:
            base = 5
        elif marked and irregular and not large:
            base = 4
        elif moderate:
            base = 3
        else:
            base = 2

    if epe and (marked or large):
        base = min(5, base + 1)

    return base


def build_rationale_text(case_id, zone, size_cc, z_score, shape_sphericity, epe, pirads):
    if size_cc is None:
        return (f"Case {case_id}: no lesion segmented within the prostate gland. "
                f"No suspicious focus identified; PI-RADS category 1 (routine follow-up).")

    zone_txt = {"PZ": "peripheral zone", "TZ": "transition zone", "mixed": "gland (PZ/TZ boundary)"}[zone]
    if shape_sphericity is None:
        margin_txt = "indeterminate (lesion too small for a reliable margin estimate)"
    elif shape_sphericity < IRREGULAR_SPHERICITY:
        margin_txt = f"irregular, ill-defined (sphericity={shape_sphericity:.2f})"
    else:
        margin_txt = f"round to oval, more circumscribed (sphericity={shape_sphericity:.2f})"
    intensity_txt = (
        "markedly hypointense" if z_score <= MARKED_HYPOINTENSE_Z else
        "moderately hypointense" if z_score <= MODERATE_HYPOINTENSE_Z else
        "mildly hypointense/indeterminate"
    )
    epe_txt = (
        " A substantial portion of the lesion falls outside the automated whole-gland "
        "segmentation, suggestive of extraprostatic extension (note: the gland mask is "
        "AI-derived and not verified at the capsule boundary, so this may partly reflect "
        "gland-segmentation imprecision rather than confirmed extension)."
        if epe else ""
    )
    dce_txt = (
        " Note: this dataset is biparametric (no DCE sequence available); real PI-RADS "
        "v2.1 would use DCE as a tie-breaker for a peripheral-zone category-3 lesion "
        "(early enhancement can upgrade PZ-3 to PZ-4) -- that assessment cannot be made "
        "here, so this category should be treated as a lower bound, not a confirmed 3."
        if (zone == "PZ" and pirads == 3) else ""
    )
    return (
        f"Case {case_id}: a {size_cc:.2f} cc lesion in the {zone_txt} is {intensity_txt} on T2W "
        f"(z={z_score:.2f} relative to same-zone background tissue) with {margin_txt} margins.{epe_txt} "
        f"These measurements are consistent with PI-RADS category {pirads}.{dce_txt}"
    )


def synthesize_rationale(case_id, arrs, spacing_zyx, voxel_vol, marksheet_pirads):
    lesion = arrs["lesion"] > 0
    gland = arrs["gland"] > 0
    zone_arr = arrs["zone"]
    t2w = arrs["t2w"].astype(np.float32)

    result = {"case_id": case_id, "lesion_present": bool(lesion.any()), "marksheet_pirads": marksheet_pirads}

    if not lesion.any():
        result.update({
            "lesion_volume_mm3": 0.0, "lesion_volume_cc": 0.0,
            "zone_location": "none", "zone_pz_fraction": None, "zone_tz_fraction": None,
            "t2w_intensity_zscore": None, "shape_sphericity": None,
            "extraprostatic_extension": False, "frac_lesion_outside_gland": None,
            "predicted_pirads": 1,
        })
        result["rationale_text"] = build_rationale_text(case_id, None, None, None, None, False, 1)
        return result

    lesion_vol_mm3 = float(lesion.sum() * voxel_vol)

    n_pz = int((lesion & (zone_arr == PZ_LABEL)).sum())
    n_tz = int((lesion & (zone_arr == TZ_LABEL)).sum())
    n_zoned = n_pz + n_tz
    pz_frac = n_pz / n_zoned if n_zoned > 0 else 0.0
    tz_frac = n_tz / n_zoned if n_zoned > 0 else 0.0
    if n_zoned == 0:
        zone_loc = "mixed"
    elif pz_frac >= 0.65:
        zone_loc = "PZ"
    elif tz_frac >= 0.65:
        zone_loc = "TZ"
    else:
        zone_loc = "mixed"

    dominant_zone_label = PZ_LABEL if pz_frac >= tz_frac else TZ_LABEL
    same_zone_ref = gland & (zone_arr == dominant_zone_label) & ~lesion
    ref_mask = same_zone_ref if same_zone_ref.sum() >= 50 else (gland & ~lesion)
    lesion_mean = t2w[lesion].mean()
    ref_mean = t2w[ref_mask].mean()
    ref_std = t2w[ref_mask].std()
    z_score = float((lesion_mean - ref_mean) / ref_std) if ref_std > 1e-6 else 0.0

    shape_sph = sphericity(lesion, spacing_zyx)

    n_outside = int((lesion & ~gland).sum())
    frac_outside = n_outside / int(lesion.sum())
    epe = bool(frac_outside > EPE_FRACTION_THRESHOLD)

    pirads = classify_pirads(zone_loc if zone_loc != "mixed" else "TZ", lesion_vol_mm3, z_score, shape_sph, epe)

    result.update({
        "lesion_volume_mm3": round(lesion_vol_mm3, 1),
        "lesion_volume_cc": round(lesion_vol_mm3 / 1000.0, 3),
        "zone_location": zone_loc,
        "zone_pz_fraction": round(pz_frac, 3),
        "zone_tz_fraction": round(tz_frac, 3),
        "t2w_intensity_zscore": round(z_score, 3),
        "shape_sphericity": round(shape_sph, 3) if shape_sph else None,
        "extraprostatic_extension": epe,
        "frac_lesion_outside_gland": round(frac_outside, 3),
        "predicted_pirads": pirads,
    })
    result["rationale_text"] = build_rationale_text(
        case_id, zone_loc if zone_loc != "mixed" else "mixed",
        result["lesion_volume_cc"], z_score, shape_sph, epe, pirads,
    )
    return result


# ================================= MAIN =====================================


def load_all_cases(root: Path):
    with open(root / "selected_cases.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    root = find_dataset_root()
    print(f"Dataset root: {root}")
    cases = load_all_cases(root)
    n_pos_total = sum(1 for c in cases if c["case_csPCa"] == "YES")
    n_neg_total = sum(1 for c in cases if c["case_csPCa"] == "NO")
    print(f"Loaded {len(cases)} cases from selected_cases.csv "
          f"({n_pos_total} positive, {n_neg_total} negative)")

    RATIONALES_DIR.mkdir(parents=True, exist_ok=True)

    n_excluded = 0
    n_written_pos = 0
    n_written_neg = 0
    exclusions = []

    for i, row in enumerate(cases):
        case_id = row["case_id"]
        is_positive = row["case_csPCa"] == "YES"

        try:
            aligned = load_aligned_case(root / "images", root / "labels", case_id)
            arrs = aligned_case_to_arrays(aligned)
            spacing_xyz = aligned["t2w"].GetSpacing()
            spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
            voxel_vol = voxel_volume_mm3(aligned["t2w"])

            passed, defect_details = run_defect_check(case_id, is_positive, arrs, spacing_zyx, voxel_vol)
            if not passed:
                failed_checks = [k for k, v in defect_details["checks"].items() if not v]
                print(f"  [{i+1}/{len(cases)}] {case_id}  EXCLUDED (defect check failed: "
                      f"{', '.join(failed_checks)})  detail={defect_details}")
                exclusions.append({"case_id": case_id, "case_csPCa": row["case_csPCa"],
                                    "reason": failed_checks, "detail": defect_details})
                n_excluded += 1
                continue

            marksheet_pirads = row.get("lesion_PIRADS") or None
            result = synthesize_rationale(case_id, arrs, spacing_zyx, voxel_vol, marksheet_pirads)
            result["defect_check"] = defect_details

            with open(RATIONALES_DIR / f"{case_id}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            if is_positive:
                n_written_pos += 1
            else:
                n_written_neg += 1
            print(f"  [{i+1}/{len(cases)}] {case_id}  OK  predicted_pirads={result['predicted_pirads']}  "
                  f"lesion_present={result['lesion_present']}")

        except Exception as e:
            print(f"  [{i+1}/{len(cases)}] {case_id}  ERROR: {e}")
            exclusions.append({"case_id": case_id, "case_csPCa": row["case_csPCa"],
                                "reason": ["exception"], "detail": str(e)})
            n_excluded += 1

    with open(KAGGLE_WORKING / "exclusions.json", "w", encoding="utf-8") as f:
        json.dump(exclusions, f, indent=2)

    print("\n" + "=" * 78)
    print("DIRECTION A SUMMARY")
    print("=" * 78)
    print(f"Total cases in selected_cases.csv: {len(cases)}  "
          f"({n_pos_total} positive, {n_neg_total} negative)")
    print(f"Excluded as likely defects / errors:  {n_excluded}")
    print(f"Final usable rationales written:      {n_written_pos + n_written_neg}  "
          f"({n_written_pos} positive, {n_written_neg} negative)")
    print(f"Rationales dir: {RATIONALES_DIR}")
    print(f"Exclusions log: {KAGGLE_WORKING / 'exclusions.json'}")


if __name__ == "__main__":
    main()
