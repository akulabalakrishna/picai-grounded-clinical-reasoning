# Findings — Data Quality Defects in picai_labels (Direction A/D evidence)

> **Reconstruction notice (provenance):** The original `findings.md` was accidentally deleted
> during a filesystem reorganization on 2026-09-24. This version was rebuilt the same day
> directly from `data/sanity_check_report.csv` (the sanity-check output that these findings
> describe) — not from memory. Every claim below is backed by the exact rows quoted.

## Summary

Of the **220 candidate positive cases** considered for the final 421-case package, **19 (8.6%)**
failed an automated mask-alignment sanity check and were excluded before packaging
(`03_sanity_check.py`). These are **real annotation defects in the upstream `picai_labels`
dataset**, not artifacts of this project's own extraction/geometry pipeline — this is the basis
for citing this defect rate as supporting evidence for Direction D's trustworthiness thesis
("grounding/verification matters even for expert-adjacent annotation pipelines").

## The evidence

`data/sanity_check_report.csv` runs four independent checks per candidate case:
`check_geometry_aligned`, `check_gland_nonempty`, `check_lesion_matches_label`, and
`check_lesion_mostly_in_gland`. For **all 19 failed cases**, the failure is isolated to exactly
one check:

- `check_geometry_aligned` = **True** for all 19 (the lesion mask shares the correct voxel grid —
  this rules out a registration/resampling bug in this project's own pipeline)
- `check_gland_nonempty` = **True** for all 19
- `check_lesion_matches_label` = **True** for all 19
- `check_lesion_mostly_in_gland` = **False** for all 19 — this is the sole failure mode

In other words: the lesion mask is correctly aligned to the same grid as the gland mask, but is
drawn mostly **outside** the gland it's supposed to be inside — a content/annotation defect, not
a geometry/pipeline defect. `frac_lesion_in_gland` for the 19 failed cases ranges from **0.002 to
0.47** (i.e., as little as 0.2% of the lesion mask actually overlaps the gland mask):

| case_id | gland_voxels | lesion_voxels | frac_lesion_in_gland |
|---|---|---|---|
| 10078_1000078 | 59,222 | 367 | 0.063 |
| 10294_1000300 | 160,300 | 1,142 | 0.423 |
| 11442_1001466 | 88,148 | 5,852 | 0.465 |
| 10168_1000171 | 118,645 | 5,968 | 0.264 |
| 10626_1000640 | 111,871 | 3,584 | 0.332 |
| 10895_1000911 | 36,843 | 806 | 0.325 |
| 10110_1000110 | 117,100 | 3,824 | **0.007** |
| 10605_1000619 | 138,999 | 849 | **0.002** |
| 10768_1000784 | 70,436 | 2,941 | 0.470 |
| 11229_1001252 | 56,710 | 640 | 0.431 |
| 11456_1001480 | 38,952 | 865 | 0.030 |
| 10202_1000206 | 204,271 | 1,932 | (n/a) |
| 10549_1000561 | 100,131 | 8,500 | 0.312 |
| 10687_1000703 | 293,462 | 1,624 | 0.177 |
| 11050_1001070 | 184,476 | 3,472 | (n/a) |
| 11051_1001071 | 32,780 | 912 | 0.202 |
| 10355_1000361 | 48,341 | 2,257 | 0.158 |
| 10482_1000490 | 189,778 | 2,127 | 0.052 |
| 11465_1001489 | 161,783 | 3,320 | 0.382 |

(Two rows have no computable `frac_lesion_in_gland` value in the source CSV — recorded as-is.)

## Why this counts as a genuine annotation defect, not a pipeline bug

Because `check_geometry_aligned` passes for all 19 (same voxel grid, same spacing/origin as the
gland mask) while `check_lesion_mostly_in_gland` fails, the defect can't be explained by a
resampling/alignment mistake in this project's own code — the two masks are unambiguously on the
same grid, and the lesion annotation is simply centered on the wrong location relative to the
gland. This is consistent with a real annotation-pipeline defect upstream in `picai_labels`
(AI-derived gland/zone masks, human-expert lesion masks — a mismatch between the two can arise
from the lesion being marked relative to a different gland boundary than the one shipped).

## Framing for Direction D

At the project's scale (421 cases, no spare human-expert-verified positive case available to
swap in for an excluded one), there was no way to manually replace these 19 cases — they were
dropped and are documented here rather than silently absent. Every excluded case's QC overlay PNG
was kept in `data/sanity_check_overlays/` (not deleted) specifically so the failure remains
visible rather than just logged.

This is used as supporting evidence for Direction D's core thesis: **even an "expert-adjacent"
annotation pipeline (human-expert lesion masks laid over AI-derived gland/zone masks) has a
real, non-trivial defect rate (8.6% here) that automated verification catches and a downstream
consumer would otherwise silently inherit.** It's a concrete instance of the same failure mode
Direction D's calibration/robustness/consistency checks are designed to surface in model
predictions, but found instead in the "ground truth" itself.

---

*Numbers in this document were read directly from `data/sanity_check_report.csv` on 2026-09-24.*
