"""
Selects a real, mixed-label sample from picai/manifest.csv and materializes
it under E:\\Cancer_IITH\\picai\\images and \\labels.

STRICT DATA POLICY (default, no flag needed): the primary/core training set
uses ONLY the 220 csPCa-positive cases with a REAL human-expert lesion
delineation (picai_labels/csPCa_lesion_delineations/human_expert/resampled)
-- not the other 205 csPCa-positive cases in the dataset, which only have
an AI-derived (Bosma22a) lesion mask. "case_csPCa == YES" alone (425 cases
total) is NOT sufficient for the primary results; it must also have
has_lesion_human == True in the manifest.

NOTE -- 220 is the ELIGIBLE pool size, not the final trained-on count:
03_sanity_check.py (run after this script) automatically excludes any case
whose lesion mask fails mask-alignment verification. In practice 19 of
these 220 (8.6%) fail that check (lesion mask centroid far outside the
gland mask despite an identical voxel grid) and get dropped before
packaging -- so the real primary positive count is 201, not 220. This
defect rate is itself a documented finding (picai/findings.md), not
something to paper over.

--include-ai-derived-masks unlocks the full 425-case pool (220 real +
205 AI-derived) for the SEPARATE "pragmatic" comparison run planned for
Week 3-4 -- every case pulled from the AI-derived-only subset is tagged
lesion_mask_source=ai_derived in selected_cases.csv so it can always be
filtered out or reported separately; this flag must never be used for the
primary/core results.

Negative cases (case_csPCa == NO) are unaffected by this policy -- "no
lesion" is real, exact ground truth regardless of source (there's no
AI-derived vs human-expert distinction for an empty mask).

Usage:
  # Primary/core run: ALL 220 real-verified positives + matching negatives
  python 02_select_and_extract.py

  # Take fewer than all available positives (e.g. for a faster iteration)
  python 02_select_and_extract.py --n-positive 30

  # Explicit negative count (default: match n-positive, i.e. balanced)
  python 02_select_and_extract.py --n-negative 100

  # Week 3-4 "pragmatic" run only -- includes AI-derived-mask positives
  python 02_select_and_extract.py --include-ai-derived-masks --n-positive 425

Selection policy within whichever pool is in play:
  - If n is None or >= the eligible pool size, ALL eligible cases are taken
    (order is then irrelevant -- this is what happens by default at n=220,
    since that's the entire real-verified pool).
  - Otherwise, spreads picks round-robin across `center` values (sorted by
    patient_id within each center) for site diversity in a smaller sample.
    This is prefix-stable: growing n reuses the same earlier picks rather
    than reshuffling, so e.g. the original 15-case pilot's cases are a
    subset of any larger n drawn from the same pool.

For each selected case, copies from the fold zip (targeted extraction, not
a full unzip) and from picai_labels:
  picai/images/<pid>_<sid>/<pid>_<sid>_t2w.mha
  picai/images/<pid>_<sid>/<pid>_<sid>_adc.mha
  picai/images/<pid>_<sid>/<pid>_<sid>_hbv.mha
  picai/labels/<pid>_<sid>/<pid>_<sid>_gland.nii.gz
  picai/labels/<pid>_<sid>/<pid>_<sid>_zone.nii.gz
  picai/labels/<pid>_<sid>/<pid>_<sid>_lesion.nii.gz   (real human-expert
                                                          mask, real
                                                          AI-derived mask
                                                          [only with
                                                          --include-ai-
                                                          derived-masks],
                                                          or real all-zero
                                                          for negatives)

Run:
  E:\\Cancer_IITH\\.venv\\Scripts\\python.exe E:\\Cancer_IITH\\picai\\scripts\\02_select_and_extract.py [flags]
"""
import argparse
import csv
import shutil
import zipfile
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(r"E:\Cancer_IITH")
LABELS = ROOT / "picai_labels"
MANIFEST = ROOT / "picai" / "manifest.csv"
IMAGES_OUT = ROOT / "picai" / "images"
LABELS_OUT = ROOT / "picai" / "labels"
SELECTION_OUT = ROOT / "picai" / "selected_cases.csv"

GLAND_DIR = LABELS / "anatomical_delineations" / "whole_gland" / "AI" / "Bosma22b"
ZONE_DIR = LABELS / "anatomical_delineations" / "zonal_pz_tz" / "AI" / "HeviAI23"
LESION_HUMAN_DIR = LABELS / "csPCa_lesion_delineations" / "human_expert" / "resampled"
LESION_AI_DIR = LABELS / "csPCa_lesion_delineations" / "AI" / "Bosma22a"


def load_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def spread_by_center(rows, n):
    """Round-robin across `center` values so a smaller-than-all sample isn't
    all one site. If n >= len(rows), returns all of rows (order irrelevant)."""
    if n >= len(rows):
        return list(rows)
    by_center = {}
    for r in rows:
        by_center.setdefault(r["center"], []).append(r)
    for lst in by_center.values():
        lst.sort(key=lambda r: r["patient_id"])
    centers = sorted(by_center.keys())
    picked = []
    i = 0
    while len(picked) < n and any(by_center[c] for c in centers):
        c = centers[i % len(centers)]
        if by_center[c]:
            picked.append(by_center[c].pop(0))
        i += 1
    return picked[:n]


def select_cases(rows, include_ai_derived: bool, n_positive, n_negative):
    if include_ai_derived:
        positive_pool = [r for r in rows if r["case_csPCa"] == "YES"]
    else:
        positive_pool = [r for r in rows if r["case_csPCa"] == "YES" and r["has_lesion_human"] == "True"]
    negative_pool = [r for r in rows if r["case_csPCa"] == "NO"]

    n_pos = len(positive_pool) if n_positive is None else n_positive
    n_neg = n_pos if n_negative is None else n_negative  # default: balanced

    pos_sel = spread_by_center(positive_pool, n_pos)
    neg_sel = spread_by_center(negative_pool, n_neg)
    return pos_sel, neg_sel


def extract_image(zip_path, arcname, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        with z.open(arcname) as src, open(dest_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def write_zero_mask(dest_path, reference_image_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ref = sitk.ReadImage(str(reference_image_path))
    zeros = sitk.Image(ref.GetSize(), sitk.sitkUInt8)
    zeros.CopyInformation(ref)
    sitk.WriteImage(zeros, str(dest_path))


def copy_mask(src_path, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dest_path)


def resolve_lesion_mask(case_id, is_positive, include_ai_derived, dest_path, t2w_path):
    """Returns the lesion_mask_source tag: 'human_expert', 'ai_derived', or 'zero_negative'."""
    if not is_positive:
        write_zero_mask(dest_path, t2w_path)
        return "zero_negative"

    human_path = LESION_HUMAN_DIR / f"{case_id}.nii.gz"
    if human_path.exists():
        copy_mask(human_path, dest_path)
        return "human_expert"

    if include_ai_derived:
        ai_path = LESION_AI_DIR / f"{case_id}.nii.gz"
        copy_mask(ai_path, dest_path)
        return "ai_derived"

    raise RuntimeError(
        f"{case_id} is positive with no human-expert lesion mask and "
        f"--include-ai-derived-masks is not set -- this case should never "
        f"have been selected under the strict policy (bug in select_cases?)."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-ai-derived-masks", action="store_true",
                         help="Unlock the full 425-case positive pool (220 real + 205 "
                              "AI-derived). NEVER use for primary/core results -- only "
                              "for the labeled Week 3-4 'pragmatic' comparison run.")
    parser.add_argument("--n-positive", type=int, default=None,
                         help="Number of positive cases to select. Default: ALL eligible "
                              "in the chosen pool (220 under the strict policy, up to 425 "
                              "with --include-ai-derived-masks).")
    parser.add_argument("--n-negative", type=int, default=None,
                         help="Number of negative cases to select. Default: match "
                              "n-positive (balanced set). This default is a judgment "
                              "call, not a locked decision -- override once you get to "
                              "the scaling-math conversation.")
    args = parser.parse_args()

    rows = load_manifest()
    pos_sel, neg_sel = select_cases(rows, args.include_ai_derived_masks, args.n_positive, args.n_negative)
    selected = pos_sel + neg_sel

    policy_txt = "PRAGMATIC (includes AI-derived masks)" if args.include_ai_derived_masks else "STRICT (real human-expert masks only)"
    print(f"Policy: {policy_txt}")
    print(f"Selected {len(selected)} cases ({len(pos_sel)} positive, {len(neg_sel)} negative)")

    log_rows = []
    for r in selected:
        pid, sid = r["patient_id"], r["study_id"]
        case_id = f"{pid}_{sid}"
        zip_path = ROOT / r["zip_file"]
        is_positive = r["case_csPCa"] == "YES"

        # --- images ---
        with zipfile.ZipFile(zip_path) as z:
            arcnames = {Path(n).name: n for n in z.namelist() if not n.endswith("/")}
        for seq in ("t2w", "adc", "hbv"):
            arcname = arcnames[f"{case_id}_{seq}.mha"]
            dest = IMAGES_OUT / case_id / f"{case_id}_{seq}.mha"
            extract_image(zip_path, arcname, dest)

        t2w_path = IMAGES_OUT / case_id / f"{case_id}_t2w.mha"

        # --- masks ---
        gland_dest = LABELS_OUT / case_id / f"{case_id}_gland.nii.gz"
        zone_dest = LABELS_OUT / case_id / f"{case_id}_zone.nii.gz"
        lesion_dest = LABELS_OUT / case_id / f"{case_id}_lesion.nii.gz"

        copy_mask(GLAND_DIR / f"{case_id}.nii.gz", gland_dest)
        copy_mask(ZONE_DIR / f"{case_id}.nii.gz", zone_dest)
        lesion_source = resolve_lesion_mask(
            case_id, is_positive, args.include_ai_derived_masks, lesion_dest, t2w_path
        )

        print(f"  {case_id}  csPCa={r['case_csPCa']:3s}  center={r['center']:5s}  lesion_mask={lesion_source}")
        log_rows.append({**r, "case_id": case_id, "lesion_mask_source": lesion_source})

    with open(SELECTION_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    n_ai = sum(1 for r in log_rows if r["lesion_mask_source"] == "ai_derived")
    print(f"\nWrote selection log to {SELECTION_OUT}")
    if n_ai > 0:
        print(f"NOTE: {n_ai} case(s) use an AI-derived lesion mask (lesion_mask_source="
              f"ai_derived in the CSV) -- confirm this is the intended pragmatic run, "
              f"not accidentally mixed into a primary/core run.")


if __name__ == "__main__":
    main()
