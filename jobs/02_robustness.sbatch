#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=3:00:00
#SBATCH --job-name=nhb_robust
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

module load python/miniforge3_pytorch
module list
echo "job is starting on `hostname`"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd ${REPO:?set REPO to the repository root}
python -I -m pytest -q tests/test_robustness.py tests/test_fit_pooled.py || exit 1
python -I src/robustness.py --perms 100
