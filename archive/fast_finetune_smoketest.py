import sys
sys.path.insert(0, ".")
import train_direction_b as tdb

# Shrink resolution only -- UNet3D is fully convolutional (resolution-agnostic
# as long as Y,X divisible by 8 for the three stride-2 pools). This tests the
# REAL, unmodified main()/training-loop code (contrastive pass, requires_grad
# warmup toggle, weighted backward, focal pass, checkpointing) at a fraction
# of the compute cost -- resolution doesn't change whether that control flow
# is correct, only how long a real scientific run would take.
tdb.CROP_SHAPE_ZYX = (4, 32, 32)
tdb.FINAL_SHAPE_ZYX = (4, 64, 64)

sys.argv = [
    "train_direction_b.py",
    "--data-root", "_direction_b_local_test",
    "--rationales-dir", "_direction_b_local_test/rationales",
    "--baseline-checkpoint", "_direction_b_local_test/dummy_baseline_ckpt.pt",
    "--output-dir", "_direction_b_local_test/output_finetune_fastcheck",
    "--encoder-mode", "finetune",
    "--n-epochs", "2",
    "--contrastive-warmup-epochs", "1",
    "--contrastive-group-size", "2",
    "--num-workers", "0",
]
tdb.main()
