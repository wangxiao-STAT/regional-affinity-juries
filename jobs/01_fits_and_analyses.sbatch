#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=3:00:00
#SBATCH --job-name=rv1_all
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

# Round-1 revision analyses in one job (replaces rv1_fits/robust/cf_pooled/cf_comb/design/cross, whose
# dependent jobs would have queued again after the fits): fits first, then everything else in
# parallel on the 16 cores. Each step logs to jobs/logs/rv1_<step>.log and its exit code to .exit.
# DRY=1 bash jobs/rv1_all.sbatch prints the commands without running them.

module load python/miniforge3_pytorch
module list
echo "job is starting on `hostname`"
cd ${REPO:?set REPO to the repository root}
L=jobs/logs
mkdir -p $L

threads() { export OMP_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 MKL_NUM_THREADS=$1; }
step() {        # step NAME THREADS CMD...: run CMD with its own log and exit-code file
    local name=$1 n=$2; shift 2
    if [ -n "$DRY" ]; then echo "[$name, $n threads] $*"; echo 0 > $L/rv1_$name.exit; return 0; fi
    ( threads $n; "$@" > $L/rv1_$name.log 2>&1; echo $? > $L/rv1_$name.exit )
}

threads 16
if [ -z "$DRY" ]; then python -I -m pytest -q tests/ || exit 1; fi
rm -f $L/rv1_*.exit

step robust 4 python -I src/robustness.py --only placebo_rolefe loao_women --perms 100 &
P_ROBUST=$!
step fits 12 bash -c "python -I src/fit_pooled.py && python -I src/fit_combined.py && python -I src/mechanisms.py"
if [ "$(cat $L/rv1_fits.exit)" != 0 ]; then echo "fits failed"; wait $P_ROBUST; exit 1; fi

step cf_pooled 3 bash -c "for m in M1_role M5_role_confhet M4_role_confFE; do python -I src/counterfactuals.py --model \$m --draws 1000 || exit 1; done" &
step cf_comb 4 bash -c "for m in C1_role C2_confhet C4_role_confFE C1_role_award; do python -I src/counterfactuals.py --model \$m --draws 1000 --res results/combined || exit 1; done" &
step design_m1 1 python -I src/design_sim.py --model M1_role --reps 800 --draws 40 &
step design_m5 1 python -I src/design_sim.py --model M5_role_confhet --reps 800 --draws 40 &
step cross 1 bash -c "for m in C1_role C2_confhet; do python -I src/cross_jury.py --model \$m --reps 800 --draws 40 || exit 1; done" &
wait

status=0
for f in $L/rv1_*.exit; do
    echo "$(basename $f .exit): $(cat $f)"
    [ "$(cat $f)" = 0 ] || status=1
done
exit $status
