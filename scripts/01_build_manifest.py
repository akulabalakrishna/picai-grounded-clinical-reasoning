"""
Scans the 5 PI-CAI fold zips (E:\\Cancer_IITH\\picai_public_images_fold{0..4}.zip)
and the picai_labels repo (E:\\Cancer_IITH\\picai_labels) to build a manifest of
every case that has:
  - all three imaging sequences (t2w, adc, hbv)
  - a whole-gland mask
  - a zonal (PZ/TZ) mask
  - marksheet clinical info (case_csPCa label)

Does NOT extract anything yet -- just peeks at zip file listings and label
folder listings. Output: E:\\Cancer_IITH\\picai\\manifest.csv

Run from the local venv:
  E:\\Cancer_IITH\\.venv\\Scripts\\python.exe E:\\Cancer_IITH\\picai\\scripts\\01_build_manifest.py
"""
import csv
import zipfile
from pathlib import Path

ROOT = Path(r"E:\Cancer_IITH")
LABELS = ROOT / "picai_labels"
OUT_CSV = ROOT / "picai" / "manifest.csv"

FOLD_ZIPS = [ROOT / f"picai_public_images_fold{i}.zip" for i in range(5)]

GLAND_DIR = LABELS / "anatomical_delineations" / "whole_gland" / "AI" / "Bosma22b"
ZONE_DIR = LABELS / "anatomical_delineations" / "zonal_pz_tz" / "AI" / "HeviAI23"
LESION_HUMAN_DIR = LABELS / "csPCa_lesion_delineations" / "human_expert" / "resampled"
LESION_AI_DIR = LABELS / "csPCa_lesion_delineations" / "AI" / "Bosma22a"
MARKSHEET = LABELS / "clinical_information" / "marksheet.csv"


def scan_zip_images():
    """Returns dict[(patient_id, study_id)] -> {'zip': path, 'seqs': {'t2w': arcname, ...}}"""
    cases = {}
    for zpath in FOLD_ZIPS:
        if not zpath.exists():
            print(f"WARNING: missing {zpath}")
            continue
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if name.endswith("/"):
                    continue
                fname = Path(name).name
                if not fname.endswith(".mha"):
                    continue
                stem = fname[: -len(".mha")]
                parts = stem.split("_")
                if len(parts) < 3:
                    continue
                seq = parts[-1]
                if seq not in ("t2w", "adc", "hbv"):
                    continue
                pid, sid = parts[0], parts[1]
                key = (pid, sid)
                entry = cases.setdefault(key, {"zip": zpath, "seqs": {}})
                entry["seqs"][seq] = name
    return cases


def load_marksheet():
    info = {}
    with open(MARKSHEET, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["patient_id"], row["study_id"])
            info[key] = row
    return info


def label_exists(directory, pid, sid):
    return (directory / f"{pid}_{sid}.nii.gz").exists()


def main():
    print("Scanning zip listings (no extraction)...")
    image_cases = scan_zip_images()
    print(f"Found {len(image_cases)} (patient_id, study_id) pairs with >=1 sequence file")

    marksheet = load_marksheet()
    print(f"Marksheet has {len(marksheet)} rows")

    rows = []
    for (pid, sid), entry in image_cases.items():
        seqs = entry["seqs"]
        has_all_seqs = all(s in seqs for s in ("t2w", "adc", "hbv"))
        if not has_all_seqs:
            continue

        has_gland = label_exists(GLAND_DIR, pid, sid)
        has_zone = label_exists(ZONE_DIR, pid, sid)
        has_lesion_human = label_exists(LESION_HUMAN_DIR, pid, sid)
        has_lesion_ai = label_exists(LESION_AI_DIR, pid, sid)

        clin = marksheet.get((pid, sid), {})
        case_cspca = clin.get("case_csPCa", "")

        if not (has_gland and has_zone):
            continue  # need at least gland+zone to build the 3-mask input

        rows.append(
            {
                "patient_id": pid,
                "study_id": sid,
                "zip_file": entry["zip"].name,
                "case_csPCa": case_cspca,
                "has_lesion_human": has_lesion_human,
                "has_lesion_ai": has_lesion_ai,
                "center": clin.get("center", ""),
                "prostate_volume": clin.get("prostate_volume", ""),
                "patient_age": clin.get("patient_age", ""),
                "psa": clin.get("psa", ""),
                "lesion_PIRADS": clin.get("lesion_PIRADS", ""),
                "lesion_ISUP": clin.get("lesion_ISUP", ""),
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["case_csPCa"] == "YES")
    n_pos_with_mask = sum(1 for r in rows if r["case_csPCa"] == "YES" and r["has_lesion_human"])
    n_neg = sum(1 for r in rows if r["case_csPCa"] == "NO")
    print(f"Wrote {len(rows)} eligible cases to {OUT_CSV}")
    print(f"  csPCa=YES: {n_pos}  (with human-expert lesion mask: {n_pos_with_mask})")
    print(f"  csPCa=NO:  {n_neg}")


if __name__ == "__main__":
    main()
