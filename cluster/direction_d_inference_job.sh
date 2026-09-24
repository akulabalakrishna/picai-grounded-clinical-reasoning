#!/bin/sh
#SBATCH --job-name=picai_direction_d_inference
#SBATCH --partition=cse-ai-gpu-all
#SBATCH --nodelist=dgx-v100-01
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=picai_direction_d_inference_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate picai

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Hostname=$(hostname -s)"
nvidia-smi

# Inference only -- no training, no gradients. ~421 clean forward passes
# (calibration val subset + full-dataset mask-rationale consistency) plus
# ~20*3 perturbed forward passes for robustness. At a few seconds/case on
# a V100 this should finish in well under the 4h budget above; the budget
# is generous headroom, not an expected runtime.
python direction_d_inference.py \
    --data-root ~/picai_data \
    --rationales-dir ~/picai_data/rationales \
    --baseline-checkpoint ~/picai_outputs/run1/checkpoints/step_0005040_epoch_0059_end.pt \
    --output-dir ~/picai_direction_d_outputs \
    --n-robustness-cases 20
