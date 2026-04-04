"""
Merge calibration batch files (.npz) into final maps and time series.

Usage:
    # Simple: just crop and irrigation
    python merge_batches.py olivo drip
    python merge_batches.py vite traditional
    python merge_batches.py agrumi drip

    # Auto-discover all results_* directories:
    python merge_batches.py --auto

    # Explicit directory:
    python merge_batches.py olivo drip --results-dir /path/to/results_olivo_drip_v2

Finds results directory automatically by globbing results_{crop}_{irr}*.
Output: FULL_{CROP}_{IRR}_30Y_{alpha_map,pH_map,error_map}_MERGED.tif
"""

import os
import sys
import re
import argparse
import glob
import numpy as np
import rasterio
from rasterio.transform import Affine


def find_results_dir(crop, irrigation, base_dir="."):
    """Find the results directory for a crop/irrigation combo.

    Searches for results_{crop}_{irrigation}* — handles any suffix (_30y, _v2, etc.).
    If multiple matches, picks the most recently modified one.
    """
    pattern = os.path.join(base_dir, f"results_{crop}_{irrigation}*")
    candidates = [d for d in sorted(glob.glob(pattern)) if os.path.isdir(d)]

    # Skip retry directories
    candidates = [d for d in candidates if 'retry' not in os.path.basename(d)]

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Multiple matches: pick most recently modified
    best = max(candidates, key=lambda d: os.path.getmtime(d))
    print(f"  Multiple results dirs found, using most recent: {os.path.basename(best)}")
    for d in candidates:
        marker = " <--" if d == best else ""
        print(f"    {os.path.basename(d)}{marker}")
    return best


def discover_scenarios(base_dir="."):
    """Scan for all results_{crop}_{irrigation}* directories."""
    valid_irrigations = {'drip', 'traditional', 'trad', 'rainfed'}
    valid_crops = {'vite', 'olivo', 'agrumi', 'pesco', 'grano'}

    pattern = os.path.join(base_dir, "results_*")
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))

    # Parse crop and irrigation from directory name
    dir_pattern = re.compile(r"results_(\w+?)_(drip|traditional|trad|rainfed)")

    scenarios = []
    seen = set()
    for d in dirs:
        dirname = os.path.basename(d)
        if 'retry' in dirname:
            continue
        match = dir_pattern.match(dirname)
        if match:
            crop, irrigation = match.group(1), match.group(2)
            key = (crop, irrigation)
            if crop in valid_crops and irrigation in valid_irrigations and key not in seen:
                seen.add(key)
                # Use find_results_dir to pick best if duplicates
                best_dir = find_results_dir(crop, irrigation, base_dir)
                scenarios.append({'crop': crop, 'irrigation': irrigation,
                                  'results_dir': best_dir, 'dirname': os.path.basename(best_dir)})
    return scenarios


def find_batch_files(results_dir):
    """Find all .npz batch files in results_dir.

    Searches for calibration_batch_*.npz first, falls back to any *.npz.
    """
    # Primary: calibration_batch_*.npz
    all_files = sorted(glob.glob(os.path.join(results_dir, "calibration_batch_*.npz")))

    # Fallback: any .npz that looks like a batch file
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(results_dir, "*.npz")))
        # Filter out files that are clearly not batches (e.g., timeseries, summary)
        all_files = [f for f in all_files
                     if not any(x in os.path.basename(f).lower()
                                for x in ['timeseries', 'summary', 'merged'])]

    return all_files


def deduplicate_batches(batch_files):
    """Keep only the latest file for each batch ID.

    Filename format: calibration_batch_{irrig}_{crop}_{years}y_b{BATCH_ID}_{TIMESTAMP}.npz
    Since files are sorted alphabetically and timestamp is at the end,
    later files in sorted order are newer.
    """
    batch_map = {}  # { batch_id: filepath }
    batch_pattern = re.compile(r"_b(\d+)_")

    for f in batch_files:
        match = batch_pattern.search(os.path.basename(f))
        if match:
            bid = int(match.group(1))
            # Sorted order means last match = newest timestamp
            batch_map[bid] = f
        else:
            print(f"  Warning: skipping file with unknown format: {os.path.basename(f)}")

    return sorted(batch_map.values())


def merge_scenario(results_dir, output_prefix):
    """Merge all batch .npz files for one scenario into maps + time series."""

    print(f"\n{'='*70}")
    print(f"Merging: {output_prefix}")
    print(f"  Source: {results_dir}")
    print(f"{'='*70}")

    # 1. Find batch files
    all_files = find_batch_files(results_dir)

    if not all_files:
        print(f"  No batch files found in {results_dir}")
        return False

    # 2. Deduplicate (keep latest per batch ID)
    unique_files = deduplicate_batches(all_files)

    print(f"  Found {len(all_files)} total files -> {len(unique_files)} unique batches")
    for f in unique_files:
        print(f"    -> {os.path.basename(f)}")

    # 3. Load & merge data from all batch files
    all_results = []
    ts_results = []
    failed_pixels = []

    # Load metadata from first valid file
    first_data = np.load(unique_files[0], allow_pickle=True)
    map_shape = tuple(first_data['map_shape'])
    transform = first_data['transform']
    crs_wkt = str(first_data['crs_wkt'])
    calib_years = int(first_data['calib_years']) if 'calib_years' in first_data else 30

    for f in unique_files:
        try:
            d = np.load(f, allow_pickle=True)
            if 'all_results' in d:
                all_results.extend(d['all_results'])
            if 'timeseries_results' in d:
                ts_results.extend(d['timeseries_results'])
            if 'failed_pixels' in d:
                failed_pixels.extend(d['failed_pixels'])
        except Exception as e:
            print(f"    Warning: error reading {os.path.basename(f)}: {e}")

    print(f"\n  Total pixels merged: {len(all_results)} success, {len(failed_pixels)} failed")

    if len(all_results) == 0:
        print(f"  No successful pixels to merge!")
        return False

    # 4. Create maps
    alpha_map = np.full(map_shape, np.nan, dtype=np.float32)
    ph_map = np.full(map_shape, np.nan, dtype=np.float32)
    error_map = np.full(map_shape, np.nan, dtype=np.float32)

    for res in all_results:
        if res.get('status') in ['success', 'partial_success']:
            r, c = res['pixel']
            alpha_map[r, c] = res['alpha']
            ph_map[r, c] = res['pH_mean']
            error_map[r, c] = res['error']

    # Reconstruct affine transform
    tf_array = transform
    transform_obj = Affine(tf_array[0], tf_array[1], tf_array[2],
                           tf_array[3], tf_array[4], tf_array[5])

    profile = {
        'driver': 'GTiff', 'height': map_shape[0], 'width': map_shape[1],
        'count': 1, 'dtype': np.float32, 'crs': crs_wkt,
        'transform': transform_obj, 'compress': 'lzw', 'nodata': np.nan,
    }

    # Output directory = same as results_dir parent (or current dir)
    output_dir = os.path.dirname(results_dir) if os.path.dirname(results_dir) else "."

    fname_alpha = os.path.join(output_dir, f"{output_prefix}_alpha_map_MERGED.tif")
    with rasterio.open(fname_alpha, 'w', **profile) as dst:
        dst.write(alpha_map, 1)

    fname_ph = os.path.join(output_dir, f"{output_prefix}_pH_map_MERGED.tif")
    with rasterio.open(fname_ph, 'w', **profile) as dst:
        dst.write(ph_map, 1)

    fname_error = os.path.join(output_dir, f"{output_prefix}_error_map_MERGED.tif")
    with rasterio.open(fname_error, 'w', **profile) as dst:
        dst.write(error_map, 1)

    valid_px = np.sum(~np.isnan(alpha_map))
    print(f"\n  Saved: {fname_alpha} ({valid_px} valid pixels)")
    print(f"  Saved: {fname_ph}")
    print(f"  Saved: {fname_error}")

    # 5. Save merged time series
    if ts_results:
        print(f"  Merging time series ({len(ts_results)} entries)...")

        n_days = int(calib_years * 365)

        # Collect valid time series
        valid_ts = [res for res in ts_results if res.get('success')]
        n_pixels = len(valid_ts)

        if n_pixels > 0:
            # Determine available keys from first valid result
            sample = valid_ts[0]
            ts_keys = [k for k in ['Ca_daily', 'Mg_daily', 'pH_daily', 'DIC_daily', 'v_daily']
                       if k in sample]

            arrays = {k: np.full((n_pixels, n_days), np.nan, dtype=np.float32) for k in ts_keys}
            coords = np.zeros((n_pixels, 2), dtype=np.int32)

            for i, res in enumerate(valid_ts):
                coords[i] = res['pixel']
                for k in ts_keys:
                    if k in res:
                        length = min(len(res[k]), n_days)
                        arrays[k][i, :length] = res[k][:length]

            fname_ts = os.path.join(output_dir, f"{output_prefix}_timeseries_MERGED.npz")
            np.savez_compressed(fname_ts, pixel_coords=coords, map_shape=map_shape,
                                **arrays)
            print(f"  Saved: {fname_ts} ({n_pixels} pixels, keys: {ts_keys})")
        else:
            print(f"  No valid time series to save.")
    else:
        print(f"  No time series data in batch files.")

    # 6. Print summary statistics
    alphas = [r['alpha'] for r in all_results if r.get('status') in ['success', 'partial_success']]
    if alphas:
        print(f"\n  Alpha stats: mean={np.mean(alphas):.4f}, std={np.std(alphas):.4f}, "
              f"min={np.min(alphas):.4f}, max={np.max(alphas):.4f}")

    return True


def make_output_prefix(crop, irrigation, years=30):
    """Generate output prefix like FULL_VITE_DRIP_30Y."""
    return f"FULL_{crop.upper()}_{irrigation.upper()}_{years}Y"


def main():
    parser = argparse.ArgumentParser(
        description="Merge calibration batch .npz files into final maps and time series.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_batches.py olivo drip              # finds results_olivo_drip* automatically
  python merge_batches.py vite traditional        # finds results_vite_traditional*
  python merge_batches.py agrumi drip --results-dir /path/to/custom_dir
  python merge_batches.py --auto                  # discover & merge all
        """)

    parser.add_argument('crop', nargs='?', default=None,
                        help='Crop type (vite, olivo, agrumi, pesco, grano)')
    parser.add_argument('irrigation', nargs='?', default=None,
                        help='Irrigation type (drip, traditional, rainfed)')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-discover all results_* directories and merge each')
    parser.add_argument('--years', type=int, default=30,
                        help='Calibration years for output naming (default: 30)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Explicit path to results directory (overrides auto-discovery)')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base directory to search in (default: current dir)')

    args = parser.parse_args()

    # Validate
    if not args.auto and not args.crop:
        parser.error("Provide crop and irrigation (e.g. 'olivo drip'), or use --auto")
    if args.crop and not args.irrigation and not args.auto:
        parser.error("Provide both crop and irrigation (e.g. 'olivo drip')")

    scenarios_to_merge = []

    if args.auto:
        discovered = discover_scenarios(args.base_dir)
        if not discovered:
            print(f"No results_* directories found in {os.path.abspath(args.base_dir)}")
            sys.exit(1)

        print(f"Discovered {len(discovered)} scenario(s):")
        for s in discovered:
            prefix = make_output_prefix(s['crop'], s['irrigation'], args.years)
            print(f"  {s['dirname']} -> {prefix}")
            scenarios_to_merge.append((s['results_dir'], prefix))
    else:
        crop = args.crop
        irrigation = args.irrigation

        # Find results directory
        if args.results_dir:
            results_dir = args.results_dir
        else:
            results_dir = find_results_dir(crop, irrigation, args.base_dir)

        if not results_dir or not os.path.isdir(results_dir):
            print(f"No results directory found for {crop} {irrigation}")
            print(f"  Searched: {args.base_dir}/results_{crop}_{irrigation}*")
            print(f"  Tip: use --results-dir to specify the path")
            sys.exit(1)

        prefix = make_output_prefix(crop, irrigation, args.years)
        print(f"  Directory: {results_dir}")
        print(f"  Output:    {prefix}_*_MERGED.tif")
        scenarios_to_merge.append((results_dir, prefix))

    # Run merges
    success_count = 0
    fail_count = 0

    for results_dir, prefix in scenarios_to_merge:
        ok = merge_scenario(results_dir, prefix)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    # Final summary
    print(f"\n{'='*70}")
    print(f"DONE: {success_count} merged, {fail_count} failed/empty")
    print(f"{'='*70}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
