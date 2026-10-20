#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=1:30:00
#SBATCH --job-name=rv2_all
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

# Round-2 revision runs, all in parallel: mechanism refits (extreme-effect counts), counterfactuals of the
# 2023-2025 FIFA contests under the group-share rule (R2-1), design simulations rerun with per-draw
# contenders (R2-9), and the 2023-2025 contests under the share rule and with zero-point nominees scoring
# nothing (R2-1 T2, S1), with a matched own-rule run on the same six contests (same random numbers). Each step logs to jobs/logs/rv2_<step>.log and its exit code to .exit.
module load python/miniforge3_pytorch
module list
echo "job is starting on `hostname`"
cd ${REPO:?set REPO to the repository root}
L=jobs/logs
SCORING="2023_MP 2023_WP 2024_MP 2024_WP 2025_MP 2025_WP"
threads() { export OMP_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 MKL_NUM_THREADS=$1; }
step() {
    local name=$1 n=$2; shift 2
    if [ -n "$DRY" ]; then echo "[$name, $n threads] $*"; echo 0 > $L/rv2_$name.exit; return 0; fi
    ( threads $n; "$@" > $L/rv2_$name.log 2>&1; echo $? > $L/rv2_$name.exit )
}
threads 16
if [ -z "$DRY" ]; then python -I -m pytest -q tests/ || exit 1; fi
rm -f $L/rv2_*.exit

step mech 3 python -I src/mechanisms.py &
step cf_share 6 bash -c "for m in C1_role C2_confhet; do python -I src/counterfactuals.py --model \$m --draws 1000 --res results/combined --rule share --contests $SCORING || exit 1; done" &
step design_m1 1 python -I src/design_sim.py --model M1_role --reps 800 --draws 40 &
step design_m5 1 python -I src/design_sim.py --model M5_role_confhet --reps 800 --draws 40 &
step design_share 1 bash -c "for m in M1_role M5_role_confhet; do python -I src/design_sim.py --model \$m --reps 800 --draws 40 --rule share --contests $SCORING || exit 1; done" &
step design_own6 1 bash -c "for m in M1_role M5_role_confhet; do python -I src/design_sim.py --model \$m --reps 800 --draws 40 --rule own --tag own6 --contests $SCORING || exit 1; done" &
step design_zero 1 bash -c "for m in M1_role M5_role_confhet; do python -I src/design_sim.py --model \$m --reps 800 --draws 40 --rule scoring_zero --contests $SCORING || exit 1; done" &
wait

status=0
for f in $L/rv2_*.exit; do
    echo "$(basename $f .exit): $(cat $f)"
    [ "$(cat $f)" = 0 ] || status=1
done
exit $status
