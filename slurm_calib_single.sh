#!/bin/bash
#SBATCH --job-name=calib
#SBATCH --output=calib_%j.log
#SBATCH --error=calib_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --partition=long

# ===== SET VIA --export=CROP=...,IRR=... =====
# Example: sbatch --export=CROP=olivo,IRR=drip slurm_calib_single.sh

# Map irrigation name to directory name on disk
case "$IRR" in
    traditional|trad) IRR_DIR="surface" ;;
    *)                IRR_DIR="$IRR" ;;
esac

BASEDIR="/scratch/user/lorenzo32/WATNEEDS+SMEW"
HYDRO_PATH="$BASEDIR/WB_interpolated_first4hours/${CROP}_${IRR_DIR}"
RESULTS_DIR="results_${CROP}_${IRR}_30y"

echo "=================================================="
echo "CALIBRATION: $CROP / $IRR (single job, all pixels)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Hydro: $HYDRO_PATH"
echo "Output: $RESULTS_DIR"
echo "Start: $(date)"
echo "=================================================="

cd "$BASEDIR" || exit 1

source /sw/eb/sw/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate Melone_Conda

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

mkdir -p "$RESULTS_DIR"

python calibration_full_map_multi_robust_noFert.py \
    --irrigation $IRR \
    --crop $CROP \
    --years 30 \
    --workers $SLURM_CPUS_PER_TASK \
    --hydro_dt 4h \
    --hydro_dir "$HYDRO_PATH" \
    --checkpoint-dir "$RESULTS_DIR"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
