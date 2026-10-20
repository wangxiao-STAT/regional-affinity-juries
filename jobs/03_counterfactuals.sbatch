#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=1:00:00
#SBATCH --job-name=rv1_cf2
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

# Counterfactuals again after margins moved to each award's rule (rule_percent); flips unchanged.
module load python/miniforge3_pytorch
module list
echo "job is starting on `hostname`"
cd ${REPO:?set REPO to the repository root}
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
python -I -m pytest -q tests/ || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
( for m in M1_role M5_role_confhet M4_role_confFE; do python -I src/counterfactuals.py --model $m --draws 1000 || exit 1; done ) > jobs/logs/rv1_cf2_pooled.log 2>&1 &
P1=$!
( for m in C1_role C2_confhet C4_role_confFE C1_role_award; do python -I src/counterfactuals.py --model $m --draws 1000 --res results/combined || exit 1; done ) > jobs/logs/rv1_cf2_comb.log 2>&1 &
P2=$!
wait $P1; s1=$?
wait $P2; s2=$?
echo "pooled $s1 combined $s2"
[ $s1 = 0 ] && [ $s2 = 0 ]
