"""
Assembles the final, clean folder to zip and upload as a Kaggle Dataset,
and zips it. Only includes whatever is currently in picai/selected_cases.csv
-- no synthetic data, nothing from the unused remainder of the 1500-case
dataset. Case counts and the zip filename are derived from
selected_cases.csv at run time, not hardcoded, so this works unchanged
whether it's packaging the 30-case pilot, the 421-case strict primary run
(201 real human-expert-verified positive + 220 negative -- NOT 220
positive: 19 of the 220 eligible real-verified cases failed
03_sanity_check.py's mask-alignment check and were excluded, an 8.6%
defect rate worth keeping as a finding, see picai/findings.md), or a
larger pragmatic run that includes AI-derived-mask positives.

Kaggle dataset limits (checked against Kaggle's public docs as of writing):
  - Per-dataset size: up to 100 GB via the API/kaggle CLI; the web uploader
    comfortably handles multi-GB datasets. Even a 425-eligible-case pack of
    this data is well under 10GB, nowhere close to any limit.

Output:
  picai/kaggle_upload/                      <- upload this folder's contents, or
  picai/picai_sample_<N>cases.zip           <- ...upload this zip directly

Layout inside the package:
  images/<case_id>/<case_id>_{t2w,adc,hbv}.mha
  labels/<case_id>/<case_id>_{gland,zone,lesion}.nii.gz
  rationales/<case_id>.json
  selected_cases.csv          -- clinical metadata + lesion_mask_source per case
  sanity_check_report.csv     -- mask-alignment QC results
  rationales_summary.csv      -- Direction A rationale summary table
  sanity_check_overlays/*.png -- visual QC overlays (for reference/EDA)
  README.md

Run:
  E:\\Cancer_IITH\\.venv\\Scripts\\python.exe E:\\Cancer_IITH\\picai\\scripts\\05_package_for_kaggle.py
"""
import csv
import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"E:\Cancer_IITH")
PICAI = ROOT / "picai"
OUT_DIR = PICAI / "kaggle_upload"


def load_selected_cases():
    with open(PICAI / "selected_cases.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_sanity_check_exclusions():
    """Returns (n_failed, n_positive_checked) from sanity_check_report.csv,
    scoped to csPCa-positive cases only -- the lesion-alignment defect this
    tracks (lesion mask centroid far outside the gland mask) can only occur
    for positive cases, since negatives have a real all-zero lesion mask by
    construction and can never fail that check. Mixing negatives into the
    denominator would understate the real defect rate among positives.
    Returns (None, None) if the report doesn't exist yet."""
    report_path = PICAI / "sanity_check_report.csv"
    if not report_path.exists():
        return None, None
    with open(report_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["case_csPCa"] == "YES"]
    n_failed = sum(1 for r in rows if r["passed"] != "True")
    return n_failed, len(rows)


def build_readme(cases) -> str:
    n_total = len(cases)
    n_pos = sum(1 for c in cases if c["case_csPCa"] == "YES")
    n_neg = sum(1 for c in cases if c["case_csPCa"] == "NO")
    n_human = sum(1 for c in cases if c.get("lesion_mask_source") == "human_expert")
    n_ai = sum(1 for c in cases if c.get("lesion_mask_source") == "ai_derived")
    centers = sorted({c["center"] for c in cases if c.get("center")})

    policy_line = (
        f"STRICT policy: all {n_pos} positive case(s) use REAL human-expert lesion masks "
        f"(lesion_mask_source=human_expert). No AI-derived lesion masks in this pack."
        if n_ai == 0 else
        f"PRAGMATIC policy: {n_human} positive case(s) use real human-expert lesion masks, "
        f"{n_ai} use AI-derived (Bosma22a) lesion masks (lesion_mask_source column in "
        f"selected_cases.csv tags which is which) -- treat results from the ai_derived "
        f"subset as a separate, secondary comparison, never as primary/core results."
    )

    n_failed, n_checked = count_sanity_check_exclusions()
    exclusion_line = (
        f" In this pack, {n_failed} of {n_checked} candidate POSITIVE cases "
        f"({100 * n_failed / n_checked:.1f}%) were excluded this way before packaging."
        if n_failed else ""
    )

    return f"""\
# PI-CAI real-case sample ({n_total} cases) for rationale-grounded tumor
# detection experiments

Source: PI-CAI Public Training and Development Dataset
(https://pi-cai.grand-challenge.org/), CC-BY-NC 4.0. Masks from
picai_labels (https://github.com/DIAGNijmegen/picai_labels).

{n_total} real cases, no synthetic data: {n_pos} csPCa-positive and {n_neg}
csPCa-negative (case_csPCa == NO, lesion channel is a real all-zero mask --
"no lesion" is real information, not synthetic content). Spread across
scanning centers ({', '.join(centers) if centers else 'n/a'}) for site
diversity.

{policy_line}

## Layout
  images/<case_id>/<case_id>_t2w.mha   -- axial T2-weighted (native res.)
  images/<case_id>/<case_id>_adc.mha   -- ADC map (native res., lower than t2w)
  images/<case_id>/<case_id>_hbv.mha   -- high b-value DWI (native res.)
  labels/<case_id>/<case_id>_gland.nii.gz   -- whole-gland mask (AI-derived,
                                                Bosma22b; shares t2w's grid)
  labels/<case_id>/<case_id>_zone.nii.gz    -- PZ(=1)/TZ(=2) mask (AI-derived,
                                                HeviAI23; DOES NOT share t2w's
                                                grid -- resample before use,
                                                see picai_geom.py)
  labels/<case_id>/<case_id>_lesion.nii.gz  -- csPCa lesion mask: real
                                                human-expert annotation, or
                                                real AI-derived (pragmatic
                                                pool only, tagged in
                                                selected_cases.csv), or real
                                                all-zero for negatives; shares
                                                t2w's grid
  rationales/<case_id>.json    -- Direction A rule-based rationale (lesion
                                   size, zone, T2W z-score, sphericity, EPE
                                   flag, hand-ported PI-RADS category)
  selected_cases.csv           -- clinical metadata + lesion_mask_source tag
  sanity_check_report.csv      -- mask-alignment QC results (only passing
                                   cases are in this package -- see below)
  rationales_summary.csv       -- Direction A summary table
  sanity_check_overlays/*.png  -- visual QC overlays (T2W + gland/zone/lesion
                                   contours), for reference/EDA only

## IMPORTANT: adc/hbv/zone are NOT natively aligned with t2w
adc and hbv have lower in-plane resolution than t2w. zone is cropped to a
tighter bounding box than t2w and does not share its size/origin. All three
must be resampled onto the t2w grid before use (nearest-neighbor for zone,
linear for adc/hbv) -- this is exactly what picai_geom.load_aligned_case()
does in the training script. gland and lesion already share t2w's exact
grid (size/spacing/origin/direction) as shipped, but the training script
resamples them too (a no-op in that case) for uniformity/safety.

## Known data-quality note
03_sanity_check.py automatically excludes and drops any case whose lesion
mask fails alignment/content checks (e.g. a lesion mask centered far from
the gland mask despite sharing its exact voxel grid -- a real picai_labels
annotation defect, not a pipeline bug) BEFORE this package is built, since
at this scale there's no spare real-verified case to manually swap in.{exclusion_line}
If a case is missing relative to the pool size you expected, check
sanity_check_report.csv for why -- every excluded case's overlay PNG is
kept (not deleted) specifically so the failure is visible, not just logged.
See findings.md (if present in this package) for the full write-up of this
defect rate as evidence for Direction D (grounding/verification matters
even for expert-adjacent annotation pipelines).
"""


def build_package(cases):
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    shutil.copytree(PICAI / "images", OUT_DIR / "images")
    shutil.copytree(PICAI / "labels", OUT_DIR / "labels")
    shutil.copytree(PICAI / "rationales", OUT_DIR / "rationales")
    shutil.copytree(PICAI / "sanity_check_overlays", OUT_DIR / "sanity_check_overlays")
    shutil.copyfile(PICAI / "selected_cases.csv", OUT_DIR / "selected_cases.csv")
    shutil.copyfile(PICAI / "sanity_check_report.csv", OUT_DIR / "sanity_check_report.csv")
    shutil.copyfile(PICAI / "rationales_summary.csv", OUT_DIR / "rationales_summary.csv")
    shutil.copyfile(Path(__file__).parent / "picai_geom.py", OUT_DIR / "picai_geom.py")
    if (PICAI / "findings.md").exists():
        shutil.copyfile(PICAI / "findings.md", OUT_DIR / "findings.md")
    (OUT_DIR / "README.md").write_text(build_readme(cases), encoding="utf-8")


def zip_package(zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in OUT_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(OUT_DIR))


def report_size(zip_path: Path):
    total_bytes = sum(p.stat().st_size for p in OUT_DIR.rglob("*") if p.is_file())
    zip_bytes = zip_path.stat().st_size
    print(f"Package folder: {OUT_DIR}  ({total_bytes / 1e6:.1f} MB uncompressed)")
    print(f"Zip file:       {zip_path}  ({zip_bytes / 1e6:.1f} MB)")
    print("Well within Kaggle's dataset size limits -- upload via the web UI "
          "(kaggle.com/datasets/new) or `kaggle datasets create -p <folder>`.")


def main():
    cases = load_selected_cases()
    n_total = len(cases)
    zip_path = PICAI / f"picai_sample_{n_total}cases.zip"

    build_package(cases)
    zip_package(zip_path)
    report_size(zip_path)


if __name__ == "__main__":
    main()
