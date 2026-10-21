#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=1:00:00
#SBATCH --job-name=esc_r5
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

# Eurovision juries after round 5: three groupings (literature blocs B, UN subregions E, finer C), fits,
# counterfactuals for all nine models, designs, and other national aggregation rules. One thread per
# process. Logs: jobs/logs/esc_<step>.log, exit codes in .exit.
module load python/miniforge3_pytorch
module list
echo "job is starting on `hostname`"
cd ${REPO:?set REPO to the repository root}
L=jobs/logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
step() {
    local name=$1; shift
    ( "$@" > $L/esc_$name.log 2>&1; echo $? > $L/esc_$name.exit )
}
python -I -m pytest -q tests/ || exit 1
rm -f $L/esc_*.exit
# the data step needs the downloaded Wikipedia pages (src/fetch_eurovision.py); without them the
# deposited data/processed/esc_*.csv are used as they are
if [ -d data/raw/eurovision/wikipedia ]; then
    step data bash -c "python -I src/parse_eurovision.py && python -I src/validate_eurovision.py && python -I src/build_eurovision.py"
    [ "$(cat $L/esc_data.exit)" = 0 ] || exit 1
fi
step fit python -I src/esc_analysis.py fit
[ "$(cat $L/esc_fit.exit)" = 0 ] || exit 1
for m in B1 B2 B1_top10 E1 E2 E1_top10 C1 C2 C1_top10; do
    step cf_$m python -I src/esc_analysis.py counterfactual --model $m --draws 1000 &
done
for m in B1 B2 E1 E2 C1; do
    step design_$m python -I src/esc_analysis.py design --model $m --draws 40 --reps 800 &
done
step bases_B1 python -I src/esc_analysis.py bases --model B1 --draws 1000 &
wait
status=0
for f in $L/esc_*.exit; do echo "$(basename $f .exit): $(cat $f)"; [ "$(cat $f)" = 0 ] || status=1; done
exit $status
