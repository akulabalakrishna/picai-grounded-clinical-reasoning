"""
Mask-alignment sanity check for the selected sample.

For every case in picai/selected_cases.csv:
  1. Loads t2w/adc/hbv/gland/zone/lesion and resamples everything onto the
     t2w grid via picai_geom.load_aligned_case (see that module for why
     this is required -- zone and adc/hbv do NOT natively share t2w's grid).
  2. Asserts all 6 channels now share identical size/spacing/origin.
  3. Sanity-checks content, not just geometry:
       - gland mask must be non-empty
       - positive cases (has real lesion mask) must have a non-empty
         lesion mask that mostly falls inside the gland mask
       - negative cases must have an all-zero lesion mask
  4. Saves a 3-panel overlay PNG (axial slice with the most gland+lesion
     signal) to picai/sanity_check_overlays/<case_id>.png so misalignment
     is visible, not just asserted.

Writes picai/sanity_check_report.csv summarizing pass/fail per case.

AT SCALE (this used to be a 30-case pilot where a single failure could be
manually swapped for a replacement -- at 220 cases, drawing on the ENTIRE
real human-expert-verified positive pool, there is no spare "real" case
left to swap in without violating the strict data policy). So this script
now automatically excludes failing cases rather than just reporting them:
  - picai/selected_cases.csv is rewritten to drop failed cases.
  - Their extracted image/label folders are deleted (never trainable).
  - Their overlay PNG is KEPT (not deleted) specifically so you can look at
    *why* a real case failed -- e.g. case 10110_1000110 from the original
    pilot had a lesion mask centered ~30mm from the gland mask despite an
    identical voxel grid, a genuine picai_labels annotation defect, visible
    immediately in its overlay.
  - The exclusion count matters for your reported numbers: if you started
    with all 220 real-verified positives and N fail, your primary training
    set has 220-N positives, not 220 -- report the real number.

Run:
  E:\\Cancer_IITH\\.venv\\Scripts\\python.exe E:\\Cancer_IITH\\picai\\scripts\\03_sanity_check.py
"""
import csv
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from picai_geom import load_aligned_case, aligned_case_to_arrays, assert_all_aligned

ROOT = Path(r"E:\Cancer_IITH")
IMAGES_DIR = ROOT / "picai" / "images"
LABELS_DIR = ROOT / "picai" / "labels"
SELECTED_CSV = ROOT / "picai" / "selected_cases.csv"
OVERLAYS_DIR = ROOT / "picai" / "sanity_check_overlays"
REPORT_CSV = ROOT / "picai" / "sanity_check_report.csv"


def best_slice_index(gland_arr, lesion_arr):
    """Pick the axial slice with the most gland+lesion foreground for the overlay."""
    per_slice = gland_arr.reshape(gland_arr.shape[0], -1).sum(axis=1) + \
        10 * lesion_arr.reshape(lesion_arr.shape[0], -1).sum(axis=1)
    return int(np.argmax(per_slice))


def znorm(slice_2d):
    m, s = slice_2d.mean(), slice_2d.std()
    return (slice_2d - m) / s if s > 1e-6 else slice_2d - m


def save_overlay(case_id, arrs, out_path):
    z = best_slice_index(arrs["gland"], arrs["lesion"])
    t2w_slice = znorm(arrs["t2w"][z].astype(np.float32))
    adc_slice = znorm(arrs["adc"][z].astype(np.float32))
    gland_slice = arrs["gland"][z]
    zone_slice = arrs["zone"][z]
    lesion_slice = arrs["lesion"][z]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(t2w_slice, cmap="gray")
    axes[0].set_title(f"{case_id}  T2W  slice {z}")
    axes[0].axis("off")

    axes[1].imshow(t2w_slice, cmap="gray")
    axes[1].contour(gland_slice, colors="lime", linewidths=1.2, levels=[0.5])
    axes[1].contour(zone_slice == 1, colors="cyan", linewidths=1.0, levels=[0.5])
    axes[1].contour(zone_slice == 2, colors="orange", linewidths=1.0, levels=[0.5])
    if lesion_slice.max() > 0:
        axes[1].contour(lesion_slice, colors="red", linewidths=1.5, levels=[0.5])
    axes[1].set_title("T2W + gland(green) zone(cyan/orange) lesion(red)")
    axes[1].axis("off")

    axes[2].imshow(adc_slice, cmap="gray")
    if lesion_slice.max() > 0:
        axes[2].contour(lesion_slice, colors="red", linewidths=1.5, levels=[0.5])
    axes[2].set_title("ADC (resampled to T2W grid) + lesion")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    with open(SELECTED_CSV, newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))

    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    for row in cases:
        case_id = row["case_id"]
        is_positive = row["case_csPCa"] == "YES"
        try:
            aligned = load_aligned_case(IMAGES_DIR, LABELS_DIR, case_id)
            assert_all_aligned(aligned)
            arrs = aligned_case_to_arrays(aligned)

            gland_vox = int((arrs["gland"] > 0).sum())
            lesion_vox = int((arrs["lesion"] > 0).sum())
            lesion_in_gland = int(((arrs["lesion"] > 0) & (arrs["gland"] > 0)).sum())
            frac_lesion_in_gland = lesion_in_gland / lesion_vox if lesion_vox > 0 else None

            checks = {
                "geometry_aligned": True,
                "gland_nonempty": gland_vox > 0,
                "lesion_matches_label": (lesion_vox > 0) == is_positive,
                "lesion_mostly_in_gland": (
                    frac_lesion_in_gland is None or frac_lesion_in_gland > 0.5
                ),
            }
            passed = all(checks.values())

            save_overlay(case_id, arrs, OVERLAYS_DIR / f"{case_id}.png")

            report.append({
                "case_id": case_id,
                "case_csPCa": row["case_csPCa"],
                "passed": passed,
                "gland_voxels": gland_vox,
                "lesion_voxels": lesion_vox,
                "frac_lesion_in_gland": round(frac_lesion_in_gland, 3) if frac_lesion_in_gland else "",
                **{f"check_{k}": v for k, v in checks.items()},
                "error": "",
            })
            status = "OK" if passed else "CHECK-FAILED"
            print(f"  {case_id}  {status}  gland_vox={gland_vox}  lesion_vox={lesion_vox}")

        except Exception as e:
            report.append({
                "case_id": case_id, "case_csPCa": row["case_csPCa"], "passed": False,
                "gland_voxels": "", "lesion_voxels": "", "frac_lesion_in_gland": "",
                "check_geometry_aligned": "", "check_gland_nonempty": "",
                "check_lesion_matches_label": "", "check_lesion_mostly_in_gland": "",
                "error": str(e),
            })
            print(f"  {case_id}  ERROR: {e}")

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)

    n_pass = sum(1 for r in report if r["passed"])
    print(f"\n{n_pass}/{len(report)} cases passed all sanity checks.")
    print(f"Overlays: {OVERLAYS_DIR}")
    print(f"Report:   {REPORT_CSV}")

    failed = [r for r in report if not r["passed"]]
    if failed:
        print(f"\nExcluding {len(failed)} failing case(s) from selected_cases.csv "
              f"(no spare real-verified case to swap in at this scale -- "
              f"see each overlay PNG for why it failed):")
        for r in failed:
            reason = r["error"] if r["error"] else \
                ", ".join(k for k in ("check_geometry_aligned", "check_gland_nonempty",
                                       "check_lesion_matches_label", "check_lesion_mostly_in_gland")
                          if r.get(k) is False)
            print(f"  {r['case_id']}  csPCa={r['case_csPCa']}  failed: {reason}")
            shutil.rmtree(IMAGES_DIR / r["case_id"], ignore_errors=True)
            shutil.rmtree(LABELS_DIR / r["case_id"], ignore_errors=True)

        failed_ids = {r["case_id"] for r in failed}
        kept_cases = [c for c in cases if c["case_id"] not in failed_ids]
        with open(SELECTED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(kept_cases[0].keys()))
            writer.writeheader()
            writer.writerows(kept_cases)

        n_pos_final = sum(1 for c in kept_cases if c["case_csPCa"] == "YES")
        n_neg_final = sum(1 for c in kept_cases if c["case_csPCa"] == "NO")
        print(f"\nFinal selected_cases.csv: {len(kept_cases)} cases "
              f"({n_pos_final} positive, {n_neg_final} negative) -- "
              f"use THESE numbers when reporting your training set size.")


if __name__ == "__main__":
    main()
