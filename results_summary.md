# PI-CAI Project — Results Summary

> **Reconstruction notice (provenance):** The original `results_summary.md` was accidentally
> deleted during a filesystem reorganization on 2026-09-24. This version was rebuilt the same
> day directly from the underlying verified data files still present in the project
> (`results/*.json`, `data/selected_cases.csv`, `data/sanity_check_report.csv`) — not from
> memory or estimation. Every number below is cited to its source file. Sections for which the
> underlying source file was not yet available at reconstruction time are marked **PENDING**
> rather than filled with a guess.

---

## Direction A — Rule-based rationale synthesis

Source: `data/selected_cases.csv`, `data/sanity_check_report.csv`, `data/rationales/` (421 files).

- **421 final cases** in the shipped package: **201 csPCa-positive**, **220 csPCa-negative**
  (verified by direct count of `case_csPCa` in `selected_cases.csv`).
- Lesion mask provenance: **201 cases use real human-expert lesion masks**
  (`lesion_mask_source=human_expert`), **220 negative cases carry a real all-zero mask**
  (`lesion_mask_source=zero_negative`). No AI-derived or synthetic lesion masks in the pack —
  strict policy, verified by direct count.
- **Candidate pool before exclusion:** 440 cases were sanity-checked (`sanity_check_report.csv`
  has 440 data rows); of these, **421 passed** and **19 failed**.
- All 19 exclusions were among the **220 candidate positive cases** → **19/220 = 8.6%** exclusion
  rate among positives (verified: filtering `sanity_check_report.csv` to `case_csPCa==YES` gives
  exactly 220 rows, of which 19 are `passed=False`).
- 421 rule-based rationale JSONs generated (one per final case), covering lesion presence, zone,
  T2W intensity z-score, shape sphericity, extraprostatic extension flag, and a hand-ported
  PI-RADS category.

See `findings.md` for the root-cause detail on the 19 exclusions.

## Baseline U-Net

**PENDING — not yet reconstructed.**

The training run that produced the checkpoint used elsewhere in this project
(`~/picai_outputs/run1/checkpoints/...` on the IITH cluster, referenced by
`cluster/train_direction_b_frozen_job.sh`) ran on the cluster, and its
`training_history.json` has not yet been transferred to this machine (pending `scp`, to land at
`E:\Cancer_IITH\baseline_history.json` → `results\baseline_history.json`).

For the record: this project also made several early attempts to run baseline training directly
on Kaggle (archived at `archive/kaggle_early_iterations/run1` through `run6_gputest`). All of
those Kaggle attempts (`run1`–`run5`) failed with `torch.AcceleratorError: CUDA error: no kernel
image is available for execution on the device` (a GPU/compute-capability mismatch on the Kaggle
runtime), and `run6_gputest` (a dedicated GPU-compatibility probe) never reached a conclusive
status in its last poll. None of these produced real metrics. Training was subsequently run
successfully on the IITH cluster instead — that run's real numbers will be filled in here once
`baseline_history.json` arrives.

## Direction B — Concept-grounded localization (frozen + finetune)

Source: `results/frozen_history.json` (60 logged epochs, steps 40–2400).

### Frozen-encoder run

- `val_dice` is **flatlined at exactly 0.40304948313885164 across all 60 logged epochs**
  (identical to 15 decimal places, every single epoch).
- `val_localization_accuracy` is **0.0 for all 60 epochs**.
- `contrastive_loss` **does move** over training (1.565 → 1.388, monotonically decreasing) —
  i.e., gradients are flowing and the concept encoder is training normally. The dice/localization
  flatline is therefore not obviously "the model learned nothing"; a bit-exact-identical
  validation Dice across 60 epochs while training loss changes is itself the anomaly, consistent
  with an **evaluation-path bug** (e.g., validation Dice not being recomputed against the current
  epoch's checkpoint) rather than a genuine frozen-weight collapse.
- `diagnose_localization_rank.py` was built specifically to disambiguate this: it checks whether
  `val_localization_accuracy=0.0` reflects real zero-learning or a **metric floor effect** — the
  R-precision metric's expected hit count under pure random ranking is `K²/N`, which rounds to
  `0` whenever the lesion voxel count `K` (at bottleneck resolution) is small relative to
  `N=393,216` candidate locations, even when the model's ranking is meaningfully better than
  random.
- **This project does not have a captured run output from `diagnose_localization_rank.py`** (it
  prints per-case rank diagnostics to stdout; no log of an actual run survived the deletion). The
  diagnosis above is grounded in the script's own documented purpose and the frozen_history.json
  pattern, not in a specific printed rank result.
- Cross-check: `results/calibration_result.json`'s `clean_dice_by_case` (40 cases) has a mean of
  **0.40306** — consistent with the frozen run's flatlined 0.403 figure, suggesting Direction D's
  calibration/robustness/consistency evaluation ran against this same frozen checkpoint.

### Finetune run

**PENDING** — `finetune_history.json` (source: `~/picai_outputs/direction_b_finetune/` on the
cluster) has not yet been transferred. Whether the flatline/collapse above was actually fixed by
finetuning cannot be reported until this file arrives.

## Direction D — Calibration, robustness, consistency

Source: `results/calibration_result.json`, `results/robustness_result.json`,
`results/consistency_result.json`.

### Calibration

- **ECE = 0.2178** over 2,113,929,216 total evaluated voxels.
- This headline number is misleading read in isolation. Breaking down by confidence bin: the
  three highest-count bins (bins 10–12) together hold **2,092,136,773 voxels — 98.97% of all
  voxels evaluated** — and in every one of them, **model accuracy exceeds average confidence**:

  | Bin | Voxel count | % of total | Avg. confidence | Accuracy |
  |---|---|---|---|---|
  | 10 | 34,579,264 | 1.64% | 0.704 | 0.994 |
  | 11 | 1,834,825,782 | 86.79% | 0.781 | 1.000 |
  | 12 | 222,731,727 | 10.54% | 0.806 | 1.000 |

  These are the background-dominant bins. **The model is systematically underconfident, not
  overconfident, in the regime that dominates the ECE score** — it says ~78–81% confidence where
  it is in fact correct ~99.99% of the time. This is the opposite of the "dangerously
  overconfident on wrong predictions" failure mode that ECE is usually raised to flag.
- `clean_dice_by_case`: mean **0.4031** across 40 cases (the file's own `n_val_cases` field says
  84; only 40 per-case entries are present in `clean_dice_by_case` — this discrepancy is real and
  unresolved, noted here rather than papered over).

### Robustness

- n=20 cases. `results/robustness_result.json` summary:

  | Condition | Mean Dice |
  |---|---|
  | Clean | 0.3917 |
  | Gaussian noise | 0.3810 |
  | Gaussian blur | 0.3977 |
  | Intensity shift | 0.3905 |

  Gaussian blur's mean is nominally *higher* than clean. **This is not a real robustness gain.**
  The per-case swings under blur are large and go both directions, and they largely cancel in the
  mean rather than reflecting a consistent effect:
  - Improved under blur: case `10635_1000651` (0.146 → 0.403, +0.257), case `10048_1000048`
    (0.597 → 0.793, +0.196), case `10942_1000961` (0.383 → 0.593, +0.210).
  - Degraded under blur: case `11446_1001470` (0.581 → 0.241, −0.340), case `10634_1000650`
    (0.427 → 0.311, −0.117), case `10019_1000019` (0.131 → 0.000, −0.131, i.e. total failure).

  This pattern — large bidirectional per-case variance with a roughly-canceling mean — is
  **instability under blur**, not evidence the model handles blur better than clean input.
  Noise and intensity-shift means are close to clean and show comparatively less per-case churn.

### Consistency (rationale vs. model prediction agreement)

- n=421 cases. **386/421 (91.7%) have at least one mismatch** between the Direction A rule-based
  rationale and the model's own prediction, broken down by mismatch type (a case can have more
  than one):

  | Mismatch type | Count | % of 421 |
  |---|---|---|
  | `lesion_present` | 220 | 52.3% |
  | `shape` (sphericity/irregularity) | 138 | 32.8% |
  | `zone` (PZ/TZ location) | 85 | 20.2% |

  **`n_lesion_present_mismatch=220` is exactly equal to the total number of negative cases in the
  dataset (220/220).** Spot-checking the per-case detail confirms this is not a coincidence: for
  the negative cases inspected (e.g. `10000_1000000`, `10001_1000001`, `10002_1000002`), the
  rationale correctly says `lesion_present=false` while the model predicts `lesion_present=true`
  with a nonzero predicted lesion volume in every case checked. This reads as the model
  predicting a positive lesion on essentially all negative cases — a specific, severe, and
  directly verifiable failure mode, distinct from the `zone`/`shape` mismatches (which only apply
  to cases where lesion presence already agrees or a lesion is predicted, and are lower-severity
  localization/shape disagreements rather than presence errors).

---

*Numbers in this document were read directly from the JSON/CSV files named above on 2026-09-24.
Where a source file was unavailable, the section is marked PENDING rather than estimated.*
