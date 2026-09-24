#!/bin/sh
#SBATCH --job-name=picai_direction_b_finetune
#SBATCH --partition=cse-ai-gpu-all
#SBATCH --nodelist=dgx-v100-01
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --output=picai_direction_b_finetune_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate picai

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Hostname=$(hostname -s)"
nvidia-smi

python train_direction_b.py \
    --data-root ~/picai_data \
    --rationales-dir ~/picai_data/rationales \
    --baseline-checkpoint ~/picai_outputs/run1/checkpoints/step_0005040_epoch_0059_end.pt \
    --output-dir ~/picai_outputs/direction_b_finetune \
    --encoder-mode finetune \
    --n-epochs 60 \
    --max-hours 94 \
    --lr 1e-4 \
    --num-workers 4 \
    --checkpoint-every-n-steps 20
