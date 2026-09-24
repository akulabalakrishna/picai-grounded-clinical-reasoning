# PI-CAI real-case sample (421 cases) for rationale-grounded tumor
# detection experiments

Source: PI-CAI Public Training and Development Dataset
(https://pi-cai.grand-challenge.org/), CC-BY-NC 4.0. Masks from
picai_labels (https://github.com/DIAGNijmegen/picai_labels).

421 real cases, no synthetic data: 201 csPCa-positive and 220
csPCa-negative (case_csPCa == NO, lesion channel is a real all-zero mask --
"no lesion" is real information, not synthetic content). Spread across
scanning centers (PCNN, RUMC, ZGT) for site
diversity.

STRICT policy: all 201 positive case(s) use REAL human-expert lesion masks (lesion_mask_source=human_expert). No AI-derived lesion masks in this pack.

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
at this scale there's no spare real-verified case to manually swap in. In this pack, 19 of 220 candidate POSITIVE cases (8.6%) were excluded this way before packaging.
If a case is missing relative to the pool size you expected, check
sanity_check_report.csv for why -- every excluded case's overlay PNG is
kept (not deleted) specifically so the failure is visible, not just logged.
See findings.md (if present in this package) for the full write-up of this
defect rate as evidence for Direction D (grounding/verification matters
even for expert-adjacent annotation pipelines).
