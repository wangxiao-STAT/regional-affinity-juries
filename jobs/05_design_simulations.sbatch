#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=1:00:00
#SBATCH --job-name=designs2
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

# Design simulations again: the compatriot (FIFA) and neighbour (Eurovision) bars now apply to the
# biased and the bias-free jury alike; Eurovision adds designs that remove part or all of one
# contender's regional bloc. Same seeds as before.
module load python/miniforge3_pytorch
module list
echo "job is starting on `hostname`"
cd ${REPO:?set REPO to the repository root}
L=jobs/logs
step() { local name=$1; shift; ( export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; "$@" > $L/d2_$name.log 2>&1; echo $? > $L/d2_$name.exit ); }
python -I -m pytest -q tests/ || exit 1
rm -f $L/d2_*.exit
step fifa_m1 python -I src/design_sim.py --model M1_role --reps 800 --draws 40 &
step fifa_m5 python -I src/design_sim.py --model M5_role_confhet --reps 800 --draws 40 &
wait
status=0
for f in $L/d2_*.exit; do echo "$(basename $f .exit): $(cat $f)"; [ "$(cat $f)" = 0 ] || status=1; done
exit $status
