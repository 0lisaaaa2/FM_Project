# FM_Project

## Uni BW Cluster 3
### Interactive dev job
```
salloc --partition=dev_gpu_h100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00
srun --job -pty bash
```
### Submit job
```
sbatch slurm/run_cxr_model.sh
```
### Status of Jobs

```
squeue --start
scontrol show job
```