#!/bin/bash
#SBATCH --job-name=t5_sql_ft
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

echo "=== Job started at $(date) ==="
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

mkdir -p /scratch/$USER/NLP/Fine-tuning-Language-Models/part-2/logs

singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate nlp_hw4

    cd /scratch/$USER/NLP/Fine-tuning-Language-Models/part-2

    python train_t5.py \
        --finetune \
        --learning_rate 1e-4 \
        --weight_decay 0.01 \
        --scheduler_type cosine \
        --num_warmup_epochs 2 \
        --max_n_epochs 40 \
        --patience_epochs 6 \
        --batch_size 16 \
        --test_batch_size 16 \
        --eval_every 2 \
        --num_beams 20 \
        --num_candidates 10 \
        --experiment_name experiment

    echo '=== Job finished at $(date) ==='
  "
