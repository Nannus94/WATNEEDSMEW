#!/bin/bash
#SBATCH --job-name=calib
#SBATCH --output=calib_%A_%a.log
#SBATCH --error=calib_%A_%a.err
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --partition=long

# ===== CHANGE THESE =====
CROP="olivo"
IRR="traditional"           # Python arg: drip, traditional, rainfed
# =========================

# Map irrigation name to directory name on disk
# (hydro dirs use "surface" instead of "traditional")
case "$IRR" in
    traditional|trad) IRR_DIR="surface" ;;
    *)                IRR_DIR="$IRR" ;;
esac

BASEDIR="/scratch/user/lorenzo32/WATNEEDS+SMEW"
HYDRO_PATH="$BASEDIR/WB_interpolated_first4hours/${CROP}_${IRR_DIR}"
RESULTS_DIR="results_${CROP}_${IRR}_30y"

echo "=================================================="
echo "CALIBRATION: $CROP / $IRR - Batch $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
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
    --batch-id $SLURM_ARRAY_TASK_ID \
    --checkpoint-dir "$RESULTS_DIR"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
