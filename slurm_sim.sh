#!/bin/bash
#SBATCH --job-name=ew_sim
#SBATCH --output=ew_sim_%j.log
#SBATCH --error=ew_sim_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --partition=long

# ===== CHANGE THESE =====
CROP="grano"
IRR="rainfed"
# =========================

echo "=================================================="
echo "EW SIMULATION: $CROP / $IRR"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "=================================================="

echo "[DEBUG] cd to workdir..."
cd /scratch/user/lorenzo32/WATNEEDS+SMEW || exit 1
echo "[DEBUG] cd done. Activating conda..."

source /sw/eb/sw/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate Melone_Conda
echo "[DEBUG] conda activated. Python: $(which python)"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "[DEBUG] testing imports..."
python -c "
import sys; print('[DEBUG] python ok', flush=True)
import numpy; print('[DEBUG] numpy ok', flush=True)
import pandas; print('[DEBUG] pandas ok', flush=True)
import pyEW; print('[DEBUG] pyEW ok', flush=True)
import rasterio; print('[DEBUG] rasterio ok', flush=True)
import scipy.io; print('[DEBUG] scipy ok', flush=True)
import netCDF4; print('[DEBUG] netCDF4 ok', flush=True)
print('[DEBUG] ALL IMPORTS OK', flush=True)
"
echo "[DEBUG] launching python..."
python ew_simulation.py \
    --crop $CROP \
    --irr $IRR \
    --workers $SLURM_CPUS_PER_TASK
echo "[DEBUG] python exited with code: $?"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
