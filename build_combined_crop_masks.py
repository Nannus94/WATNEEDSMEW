"""
Build combined (irrigated + rainfed) crop area maps at 10 km for Sicily.
Output: sicily10km_{crop}_total_ha.tif  (39x43, matching hydro grid)
"""

import numpy as np
import rasterio
import os

crop_dir = r"C:\Users\Latitude 5511\Desktop\WATNEEDSMEW\Aree_coltivate"

combinations = {
    "vineyard": ("sicily10km_vineyard_i_ha.tif", "sicily10km_vineyard_r_ha.tif"),
    "olives":   ("sicily10km_olives_i_ha.tif",   "sicily10km_olives_r_ha.tif"),
    "citrus":   ("sicily10km_citrus_i_ha.tif",   "sicily10km_citrus_r_ha.tif"),
    "fruits":   ("sicily10km_fruits_i_ha.tif",    "sicily10km_fruits_r_ha.tif"),
    "wheat":    (None,                             "sicily10km_wheat_r_ha.tif"),
}

for crop, (irr_file, rain_file) in combinations.items():
    with rasterio.open(os.path.join(crop_dir, rain_file)) as src:
        rain = src.read(1).astype(np.float64)
        rain[~np.isfinite(rain)] = 0.0
        profile = src.profile.copy()

    if irr_file is not None:
        with rasterio.open(os.path.join(crop_dir, irr_file)) as src:
            irr = src.read(1).astype(np.float64)
            irr[~np.isfinite(irr)] = 0.0
    else:
        irr = np.zeros_like(rain)

    total = irr + rain

    out_name = f"sicily10km_{crop}_total_ha.tif"
    out_path = os.path.join(crop_dir, out_name)
    profile.update(dtype='float32', nodata=np.nan)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(total.astype(np.float32), 1)

    n_pixels = np.sum(total > 0)
    print(f"{out_name}:")
    print(f"  Pixels with area > 0: {n_pixels}")
    print(f"  Total: {np.sum(total):,.0f} ha")
    print(f"  Irrigated: {np.sum(irr):,.0f} ha  |  Rainfed: {np.sum(rain):,.0f} ha")
    print()
