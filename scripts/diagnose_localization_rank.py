"""Diagnostic for the val_localization_accuracy=0.0000 flatline.

Loads a completed Direction B checkpoint and reports, per val
positive-with-rationale case: K (lesion voxel count at bottleneck
resolution), N (total candidate locations), and the rank of the single
best true-positive voxel among all N candidates (1 = perfect, N/2 ~=
random chance). This disambiguates "R-precision=0 because nothing was
learned" from "R-precision=0 because K/N is too small for that metric to
register real-but-imperfect localization" (R-precision's expected hits
under pure random ranking is K^2/N, which rounds to 0 whenever K is small
relative to N=393,216 -- see concept_region_localization_accuracy in
train_direction_b.py). Reuses that module's own classes/functions
directly (imported, not reimplemented) so this can't silently drift from
the real eval path. Single forward pass per case, no training/gradients.

Usage:
  python diagnose_localization_rank.py \\
      --data-root ~/picai_data \\
      --rationales-dir ~/picai_data/rationales \\
      --checkpoint ~/picai_outputs/direction_b_frozen/checkpoints/step_XXXXXXX_epoch_0059_end.pt \\
      --n-cases 6
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_direction_b import (
    UNet3D, ConceptEncoder, DirectionBDataset, direction_b_collate,
    load_case_split, rationale_to_concept_indices, autocast_ctx,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--rationales-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--embed-dim", type=int, default=64,
                   help="Must match --embed-dim used for this checkpoint's training run (default 64).")
    p.add_argument("--n-cases", type=int, default=6)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    unet = UNet3D().to(device).eval()
    unet.load_state_dict(ckpt["unet_state"])
    concept_encoder = ConceptEncoder(args.embed_dim).to(device).eval()
    concept_encoder.load_state_dict(ckpt["concept_encoder_state"])
    region_proj = nn.Linear(unet.bottleneck_channels, args.embed_dim).to(device).eval()
    region_proj.load_state_dict(ckpt["region_proj_state"])
    print(f"Loaded checkpoint: epoch={ckpt['epoch']} global_step={ckpt['global_step']} "
          f"encoder_mode={ckpt['encoder_mode']}")

    _, val_ids = load_case_split(args.data_root)
    val_ds_pos = DirectionBDataset(args.data_root / "images", args.data_root / "labels",
                                    args.rationales_dir, val_ids, positive_only=True)
    loader = DataLoader(val_ds_pos, batch_size=1, collate_fn=direction_b_collate)

    with torch.no_grad():
        for i, (x, y, case_ids, rationales) in enumerate(loader):
            if i >= args.n_cases:
                break
            case_id, rationale = case_ids[0], rationales[0]
            with autocast_ctx(device.type):
                _, bottleneck = unet(x.to(device))
            bottleneck = bottleneck[0].float()
            C, Z, Yp, Xp = bottleneck.shape
            N = Z * Yp * Xp

            mask = y[0].to(device).unsqueeze(0).unsqueeze(0).float()
            mask_ds = F.interpolate(mask, size=(Z, Yp, Xp), mode="nearest").squeeze(0).squeeze(0) > 0.5
            k = int(mask_ds.sum().item())
            if k == 0:
                print(f"{case_id}: k=0 -- lesion vanished entirely at bottleneck resolution, skipped")
                continue

            zone_idx, int_idx, shape_idx, pirads_idx, epe_idx = rationale_to_concept_indices(rationale)
            idx_t = lambda v: torch.tensor(v, device=device)
            concept_emb = concept_encoder(idx_t(zone_idx), idx_t(int_idx), idx_t(shape_idx),
                                           idx_t(pirads_idx), idx_t(epe_idx))

            flat_feat = bottleneck.permute(1, 2, 3, 0).reshape(-1, C)
            proj = F.normalize(region_proj(flat_feat), dim=-1)
            concept = F.normalize(concept_emb, dim=-1)
            sims = proj @ concept  # (N,)

            order = torch.argsort(sims, descending=True)
            true_positive_positions = mask_ds.reshape(-1)[order].nonzero(as_tuple=True)[0]
            best_rank = int(true_positive_positions.min().item()) + 1  # 1-indexed, 1 = perfect

            print(f"{case_id}: k={k}  N={N}  best_true_positive_rank={best_rank}/{N}  "
                  f"random-chance_expected_rank~={N // 2}  "
                  f"E[R-precision hits under random]={k * k / N:.5f}")


if __name__ == "__main__":
    main()
