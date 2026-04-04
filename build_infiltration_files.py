#!/usr/bin/env python
"""
Build shallow_I_{year}_{month}.mat files from WATNEEDS outputPR + outputBW.

Infiltration I = PR (effective precip) + BW (irrigation), converted to 4h steps.

Usage:
    python build_infiltration_files.py --crop vite --irrigation drip
    python build_infiltration_files.py --all

Output goes to:  infiltration_4h/{crop}_{irrigation}/shallow_I_{year}_{month}.mat
Then copy to cluster:  WB_interpolated_first4hours/{crop}_{irr_dir}/{crop}_{irr_dir}/
"""

import argparse
import os
import sys
import calendar
import numpy as np

# ---------- configuration ----------
CROP_CODES = {
    'vite': 1, 'olivo': 2, 'pesco': 3, 'agrumi': 4, 'grano': 5
}

IRR_FOLDERS = {
    'drip':        'giornalieri_DRIP_5crops',
    'traditional': os.path.join('giornalieri_SURF_5crops', 'giornalieri_SURF_5crops'),
    'rainfed':     'giornalieri_RAIN_5crops',
}

# 11 valid crop x irrigation combos
SCENARIOS = [
    ('vite',   'drip'),
    ('vite',   'traditional'),
    ('vite',   'rainfed'),
    ('olivo',  'drip'),
    ('olivo',  'traditional'),
    ('olivo',  'rainfed'),
    ('agrumi', 'drip'),
    ('agrumi', 'traditional'),
    ('pesco',  'drip'),
    ('pesco',  'traditional'),
    ('grano',  'rainfed'),
]

TARGET_SHAPE = (39, 43)  # Sicily 10km grid
STEPS_PER_DAY = 6        # 4h resolution
START_YEAR = 1983
N_YEARS = 30             # 30-year simulation


def load_mat_h5(filepath, varname):
    """Load variable from MATLAB v7.3 (.mat) file via h5py."""
    import h5py
    with h5py.File(filepath, 'r') as f:
        data = f[varname][:]  # shape from h5py: (days, 45, 41) — transposed
    return data.T  # -> (41, 45, days)


def crop_center(arr, target_rows, target_cols):
    """Center-crop array from (rows, cols, ...) to target shape."""
    r, c = arr.shape[0], arr.shape[1]
    dr = (r - target_rows) // 2
    dc = (c - target_cols) // 2
    return arr[dr:dr+target_rows, dc:dc+target_cols]


def build_scenario(base_dir, crop, irrigation, out_root):
    """Build infiltration files for one crop x irrigation scenario."""
    crop_code = CROP_CODES[crop]
    src_folder = os.path.join(base_dir, IRR_FOLDERS[irrigation])

    if not os.path.isdir(src_folder):
        print(f"  SKIP: source folder missing: {src_folder}")
        return 0

    out_dir = os.path.join(out_root, f"{crop}_{irrigation}")
    os.makedirs(out_dir, exist_ok=True)

    n_files = 0
    end_year = START_YEAR + N_YEARS

    for year in range(START_YEAR, end_year):
        for month in range(1, 13):
            pr_file = os.path.join(src_folder, f"outputPR_{year}_{month}_{crop_code}.mat")
            bw_file = os.path.join(src_folder, f"outputBW_{year}_{month}_{crop_code}.mat")

            if not os.path.isfile(pr_file):
                print(f"  WARN: missing {os.path.basename(pr_file)}")
                continue
            if not os.path.isfile(bw_file):
                print(f"  WARN: missing {os.path.basename(bw_file)}")
                continue

            # Load PR and BW (mm/day, shape after transpose: 41, 45, days)
            pr = load_mat_h5(pr_file, 'outputPR')
            bw = load_mat_h5(bw_file, 'outputBW')

            # Center-crop to (39, 43, days)
            pr = crop_center(pr, *TARGET_SHAPE)
            bw = crop_center(bw, *TARGET_SHAPE)

            # I = PR + BW (mm/day)
            I_daily = pr + bw

            # Keep only valid days for this month
            days_in_month = calendar.monthrange(year, month)[1]
            I_daily = I_daily[:, :, :days_in_month]

            # Convert to 4h steps: each step = daily_value / 6  (mm per 4h period)
            # This way load_hydro_data's *6 conversion gives back mm/day correctly
            I_4h = np.repeat(I_daily / float(STEPS_PER_DAY), STEPS_PER_DAY, axis=2)

            # Save as MATLAB v5 (compatible with scipy.io.loadmat)
            import scipy.io as sio
            out_file = os.path.join(out_dir, f"shallow_I_{year}_{month}.mat")
            sio.savemat(out_file, {'shallow_I': I_4h.astype(np.float32)},
                        do_compression=True)
            n_files += 1

    return n_files


def main():
    parser = argparse.ArgumentParser(description='Build infiltration 4h files')
    parser.add_argument('--crop', type=str, help='Crop name')
    parser.add_argument('--irrigation', type=str, help='Irrigation type')
    parser.add_argument('--all', action='store_true', help='Build all 11 scenarios')
    parser.add_argument('--base-dir', type=str, default='.', help='Project root')
    parser.add_argument('--out-dir', type=str, default='infiltration_4h', help='Output root')
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    out_root = os.path.join(base_dir, args.out_dir)

    if args.all:
        scenarios = SCENARIOS
    elif args.crop and args.irrigation:
        scenarios = [(args.crop, args.irrigation)]
    else:
        parser.error("Specify --crop and --irrigation, or use --all")

    for crop, irr in scenarios:
        print(f"\n=== {crop} / {irr} ===")
        n = build_scenario(base_dir, crop, irr, out_root)
        print(f"  -> {n} files written to {out_root}/{crop}_{irr}/")

    print("\nDone.")


if __name__ == '__main__':
    main()
