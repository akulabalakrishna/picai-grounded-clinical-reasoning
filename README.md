# PI-CAI: Grounded, Trustworthy Prostate Cancer Detection

A project on the [PI-CAI](https://pi-cai.grand-challenge.org/) prostate MRI dataset exploring
whether a lesion-detection model's predictions can be made more **interpretable** and
**verifiably trustworthy**, not just accurate. Three directions, built on a 421-case
(201 csPCa-positive / 220 csPCa-negative) real-data subset with strict provenance (no synthetic
lesion masks; real human-expert masks for every positive case).

**Full results:** [results_summary.md](results_summary.md)
**Data-quality findings:** [findings.md](findings.md)

## The three directions

- **Direction A — Rule-based rationale synthesis.** For each of the 421 cases, generates a
  structured, human-readable rationale (lesion presence, zone, size, shape, PI-RADS category)
  directly from the segmentation masks — a transparent, non-learned baseline for "why" a case is
  flagged, and the grounding signal used by Direction B.
  Script: [`scripts/kaggle_rationale_synthesis.py`](scripts/kaggle_rationale_synthesis.py)

- **Direction B — Concept-region alignment.** Trains a 3D U-Net segmentation backbone jointly
  with a contrastive concept encoder, aligning Direction A's rationale concepts (zone, shape,
  PI-RADS, etc.) to specific spatial regions in the model's bottleneck features — an attempt at
  grounding the model's internal representation in the same concepts a radiologist would cite.
  Script: [`scripts/train_direction_b.py`](scripts/train_direction_b.py) ·
  diagnostic: [`scripts/diagnose_localization_rank.py`](scripts/diagnose_localization_rank.py)

- **Direction D — Trustworthiness evaluation.** Evaluates the trained model beyond raw Dice:
  calibration (ECE, broken down by confidence bin), robustness under input perturbation
  (Gaussian noise/blur, intensity shift), and consistency between the model's predictions and
  Direction A's independent rule-based rationale.
  Script: [`scripts/kaggle_direction_d_inference.py`](scripts/kaggle_direction_d_inference.py)

See [results_summary.md](results_summary.md) for the full numbers on all three, including two
items still pending (baseline and Direction B finetune Dice — see the note at the top of that
file for why).

## Data & external artifacts

This repository contains **code, scripts, small result files, and provenance/run archives only**.
The imaging data and the packaged 421-case dataset are **not included** — they're too large for
GitHub and are hosted on Kaggle instead:

- **Dataset (421 cases, images + labels + rationales):**
  [kaggle.com/datasets/akulabalakrishna143/picai-strict-201pos-220neg](https://www.kaggle.com/datasets/akulabalakrishna143/picai-strict-201pos-220neg)
- **Dataset (baseline checkpoint):** `akulabalakrishna143/picai-baseline-checkpoint`
  (slug confirmed via `kaggle/direction_d_run1/kernel-metadata.json`'s `dataset_sources`; not
  independently verified as public since this repo never held its own `dataset-metadata.json` for
  it)
- **Kernel — Direction A rationale synthesis run:** `akulabalakrishna143/picai-direction-a-rationale-run1`
  (private Kaggle kernel — visible to the account owner only unless made public)
- **Kernel — Direction D inference run:** `akulabalakrishna143/picai-direction-d-inference-run-1`
  (private Kaggle kernel — same visibility note)

Locally, the full 4.18 GB `data/` folder (images/labels/rationales/overlays) and the 4.15 GB
`kaggle/picai_sample_421cases.zip` package exist on disk but are excluded from this repository via
`.gitignore` — see the Kaggle dataset link above for the actual data.

## Repository layout

```
results_summary.md   Full results, all three directions (some sections marked PENDING)
findings.md           picai_labels annotation-defect analysis (Direction D evidence)
requirements.txt      Frozen Python environment
scripts/              All training/inference/pipeline code
cluster/              IITH cluster sbatch job scripts
kaggle/                Kaggle dataset/kernel metadata + small run archives (rationales, logs, Direction D results)
results/               calibration/robustness/consistency JSONs + frozen-run training history
archive/               Superseded drafts and early Kaggle iteration attempts, kept for provenance
data/                  (gitignored — 421-case images/labels/overlays; see Kaggle link above)
```

## Provenance note

`results_summary.md` and `findings.md` were rebuilt on 2026-09-24 from the underlying verified
data files (`results/*.json`, `data/*.csv`) after an accidental local deletion during a filesystem
reorganization — both documents say so explicitly at the top, with every number traceable to its
source file.
