#!/bin/bash
#SBATCH --job-name=pi05_tactile_1xH200
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpuH200x8
#SBATCH --mem=180G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bgku-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 24:00:00

# ─── Environment ────────────────────────────────────────────────────────────────
module load pytorch-conda/2.8
conda activate lerobot

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi -L 2>/dev/null)"

# ─── Training (single GPU, no FSDP) ────────────────────────────────────────────
lerobot-train \
  --dataset.repo_id=Tna001/tactile_tube_200 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.type=pi05 \
  --policy.use_tactile=true \
  --policy.tactile_features='["observation.tactile.primary"]' \
  --policy.n_tactile_tokens=32 \
  --policy.tactile_feature_dim=256 \
  --policy.gradient_checkpointing=false \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.compile_model=false \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_warmup_steps=2000 \
  --policy.scheduler_decay_steps=120000 \
  --output_dir=/projects/bgku/${USER}/pi05_tactile_1gpu_h200_${SLURM_JOB_ID} \
  --job_name=pi05_tactile_1gpu_h200_${SLURM_JOB_ID} \
  --steps=120000 \
  --batch_size=32 \
  --num_workers=16 \
  --save_freq=15000 \
  --wandb.enable=true \
  --policy.repo_id=Tna001/pi05_tactile_1gpu_h200

echo "Job $SLURM_JOB_ID finished at $(date)"
