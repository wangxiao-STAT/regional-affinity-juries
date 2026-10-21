#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=0:30:00
#SBATCH --job-name=misspec
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

# Error-distribution checks: (1) ballots simulated from the fitted FIFA model with Gumbel or normal
# errors, refitted, and the estimated flip probabilities compared with the true flips (100 data sets
# per error distribution, in four seeds of 25); (2) the FIFA jury-design simulations with normal noise.
# One thread per process. Logs: jobs/logs/ms_<step>.log, exit codes in .exit.
module load python/miniforge3_pytorch
module list
echo "job is starting on `hostname`"
cd ${REPO:?set REPO to the repository root}
L=jobs/logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
step() { local name=$1; shift; ( "$@" > $L/ms_$name.log 2>&1; echo $? > $L/ms_$name.exit ); }
python -I -m pytest -q tests/ || exit 1
rm -f $L/ms_*.exit
for noise in gumbel normal; do
    for seed in 1 2 3 4; do
        step ${noise}_$seed python -I src/misspec.py --noise $noise --reps 25 --seed $seed --clock 50 &
    done
done
step design_m1 python -I src/design_sim.py --model M1_role --reps 800 --draws 40 --noise normal &
step design_m5 python -I src/design_sim.py --model M5_role_confhet --reps 800 --draws 40 --noise normal &
wait
status=0
for f in $L/ms_*.exit; do echo "$(basename $f .exit): $(cat $f)"; [ "$(cat $f)" = 0 ] || status=1; done
exit $status
