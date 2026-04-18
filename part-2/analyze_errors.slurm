#!/bin/bash
#SBATCH --job-name=t5_errors
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/errors-%j.out

echo "=== Error Analysis Job started at $(date) ==="

singularity exec \
  --overlay /scratch/$USER/overlay-dl.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate nlp_hw4
    cd /scratch/$USER/NLP/Fine-tuning-Language-Models/part-2
    python analyze_errors.py
  "
echo "=== Job finished at $(date) ==="
