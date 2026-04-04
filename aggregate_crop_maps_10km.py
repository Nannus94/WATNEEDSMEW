"""
Aggregate 1 km Italy crop area maps to the 10 km Sicily hydro grid.

Input:  Aree_coltivate/italy1km_*_ha.tif  (1254x1487, 0.00833°, all Italy)
Output: Aree_coltivate/sicily10km_*_ha.tif (39x43, 0.08333°, Sicily only)

Aggregation: SUM of hectares within each 10 km cell.
Reference grid taken from sicily_ph_cacl2_10km.tif (39x43).
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import os
import glob

base_dir = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW"
crop_dir = os.path.join(base_dir, "Aree_coltivate")

# Reference 10 km grid
ref_path = os.path.join(base_dir, "sicily_ph_cacl2_10km.tif")

with rasterio.open(ref_path) as ref:
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_shape = (ref.height, ref.width)  # (39, 43)
    ref_bounds = ref.bounds
    print(f"Reference grid: {ref_shape}, CRS={ref_crs}")
    print(f"  Bounds: {ref_bounds}")
    print(f"  Resolution: {ref.res}")

# All 1 km input files
input_files = sorted(glob.glob(os.path.join(crop_dir, "italy1km_*.tif")))
print(f"\nFound {len(input_files)} input maps.\n")

for fpath in input_files:
    fname = os.path.basename(fpath)
    # italy1km_vineyard_r_ha.tif -> sicily10km_vineyard_r_ha.tif
    out_name = fname.replace("italy1km_", "sicily10km_")
    out_path = os.path.join(crop_dir, out_name)

    with rasterio.open(fpath) as src:
        src_data = src.read(1).astype(np.float64)
        # Replace NaN with 0 for summation
        src_data[~np.isfinite(src_data)] = 0.0

        dst_data = np.zeros(ref_shape, dtype=np.float64)

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.sum,
        )

    # Write output
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': ref_shape[1],
        'height': ref_shape[0],
        'count': 1,
        'crs': ref_crs,
        'transform': ref_transform,
        'nodata': np.nan,
    }
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(dst_data.astype(np.float32), 1)

    valid = dst_data[dst_data > 0]
    total_ha = np.nansum(dst_data)
    print(f"{out_name}:")
    print(f"  Pixels with area > 0: {len(valid)}")
    print(f"  Total area: {total_ha:,.0f} ha")
    if len(valid) > 0:
        print(f"  Mean (non-zero): {np.mean(valid):,.1f} ha/pixel")
        print(f"  Max: {np.max(valid):,.1f} ha/pixel")
    print()

print("Done. All maps saved to Aree_coltivate/")
