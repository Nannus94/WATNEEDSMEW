#!/bin/bash
#SBATCH --job-name=ew_sim
#SBATCH --output=ew_%j.log
#SBATCH --error=ew_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=250G
#SBATCH --time=06:00:00

# Usage: sbatch --export=CROP=olivo,IRR=drip slurm_ew.sh

BASEDIR="/scratch/user/lorenzo32/WATNEEDS+SMEW"

echo "=================================================="
echo "EW SIMULATION: $CROP / $IRR"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "=================================================="

cd "$BASEDIR" || exit 1

source /sw/eb/sw/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate Melone_Conda

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python ew_simulation.py \
    --crop $CROP \
    --irr $IRR \
    --scenarios noEW basalt \
    --workers $SLURM_CPUS_PER_TASK \
    --rock-dose 4000.0 \
    --app-years 0 10 20 \
    --base-dir "$BASEDIR"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
