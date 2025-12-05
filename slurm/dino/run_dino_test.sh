#!/bin/bash
#SBATCH --job-name=cxr_model
#SBATCH --output=cxr_model_%j.out
#SBATCH --error=cxr_model_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load jupyter/ai/2025-02-20
source /home/st/st_us-053000/st_st193195/.venv/bin/activate
cd /home/st/st_us-053000/st_st193195/FM_Project/

# Run the script
python dino_v3.py --datasetdir /home/st/st_us-053000/st_st193195/datasets/animals/ --test

