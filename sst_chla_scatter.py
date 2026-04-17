# =============================================================================
# sst_chla_scatter.py
# =============================================================================
# PURPOSE:
#   Explore the ecological relationship between sea surface temperature and
#   phytoplankton biomass (Chlor-a) using a density-weighted scatter plot.
#
# SCIENTIFIC CONTEXT:
#   In many ocean regions a negative correlation between SST and Chlor-a
#   exists because:
#   • Cold water = upwelling → brings deep, nutrient-rich water to the surface
#                             → fuels phytoplankton growth → high Chlor-a
#   • Warm water = stratified → thermocline traps nutrients below → low Chlor-a
#   Frontal zones (boundaries between warm and cold water masses) are often
#   hotspots of biological productivity and are visible as diagonal clouds of
#   points in this scatter plot.
#   Note: in some regions (e.g. Arctic) the relationship reverses — sea ice
#   melt brings cold nutrient-poor freshwater but releases phytoplankton into
#   newly ice-free sunlit water.
#
# WHY HEXBIN AND NOT SCATTER?
#   A standard scatter plot of 5–20 million points would:
#   • Take minutes to render (each point is drawn individually)
#   • Look like a solid black mass (overplotting — millions of overlapping dots)
#   ax.hexbin() bins the points into hexagonal grid cells and colours each
#   cell by the count of points inside. This handles millions of points in
#   seconds and reveals the density structure of the distribution.
#
# WHY LOG10(CHL-A) ON THE Y-AXIS?
#   Chlor-a follows an approximately log-normal distribution — a few extreme
#   bloom values would compress the whole distribution into a narrow band at
#   the bottom of a linear scale. log10 makes the distribution much more
#   symmetric and easier to interpret visually.
#
# ECOLOGICAL THRESHOLDS (horizontal reference lines):
#   log10(1.0) = 0   → 1 mg/m³: mesotrophic/eutrophic boundary
#   log10(0.1) = -1  → 0.1 mg/m³: oligotrophic/mesotrophic boundary
#   These thresholds are widely used in ocean colour literature for
#   classifying trophic status and primary productivity estimates.
#
# CO-LOCATION STRATEGY — pyresample nearest-neighbour:
#   SST  (SLSTR, ~1 km native) and Chlor-a (OLCI, ~600 m after subsampling)
#   live on different irregular swath grids. Comparing pixel #N of the
#   flattened SST array with pixel #N of the flattened Chl-a array is
#   geometrically wrong — they map to different geographic locations.
#
#   Correct approach (implemented here):
#     1. Read the 2-D lat/lon coordinates stored in each L2 NetCDF.
#     2. Wrap each swath as a pyresample SwathDefinition.
#     3. Define a common regular 0.01° output grid covering the overlap.
#     4. Resample both swaths onto that grid with nearest-neighbour (kd-tree).
#     5. Compare pixels at the same grid cell — now truly co-located.
#
# PREREQUISITE:
#   Run sst_retrieval.py and chlora_retrieval.py first.
# =============================================================================

import glob
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pyresample import geometry, kd_tree


def find_l2_netcdf(pattern, description):
    """Auto-detect a L2 NetCDF output file by glob pattern."""
    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {description} file found matching '{pattern}' in {os.getcwd()}\n"
            f"Run sst_retrieval.py and chlora_retrieval.py first."
        )
    if len(matches) > 1:
        matches.sort(key=os.path.getmtime, reverse=True)
        print(f"Multiple {description} files found, using most recent: {matches[0]}")
    print(f"Using {description}: {matches[0]}")
    return matches[0]


def load_l2_with_coords(filepath, varname):
    """
    Load a L2 NetCDF variable together with its 2-D lat/lon coordinates.

    Returns (data, lat, lon) as 2-D float32 arrays.
    Both retrieval scripts store lat/lon as non-dimension coordinates on
    the (y, x) swath grid — they are NOT regular 1-D axis arrays.
    """
    ds = xr.open_dataset(filepath)
    data = ds[varname].values.astype(np.float32)
    lat  = ds['lat'].values.astype(np.float32)
    lon  = ds['lon'].values.astype(np.float32)
    ds.close()
    return data, lat, lon


# ---- Auto-detect L2 files ----
sst_file = find_l2_netcdf('sentinel3_SST_L2*.nc',   'SST L2')
chl_file = find_l2_netcdf('sentinel3_ChlorA_L2*.nc', 'Chlor-a L2')

sst_data, lat_sst, lon_sst = load_l2_with_coords(sst_file, 'sst')
chl_data, lat_chl, lon_chl = load_l2_with_coords(chl_file, 'chla')

print(f"SST  swath shape : {sst_data.shape}  ({sst_data.size:,} pixels)")
print(f"Chl-a swath shape: {chl_data.shape}  ({chl_data.size:,} pixels)")

# ---- Co-location via pyresample nearest-neighbour resampling ----
#
# Step 1 — wrap each irregular swath in a SwathDefinition.
#   SwathDefinition accepts 2-D lon/lat arrays, which is exactly what our
#   L2 NetCDF files contain.  No flattening needed at this stage.
sst_swath = geometry.SwathDefinition(lons=lon_sst, lats=lat_sst)
chl_swath = geometry.SwathDefinition(lons=lon_chl, lats=lat_chl)

# Step 2 — define a common regular output grid.
#   Resolution: 0.01° ≈ 1 km at mid-latitudes, matching the coarser of the
#   two products (SLSTR SST at ~1 km).  The grid covers only the geographic
#   overlap of the two swaths so no output cells are left empty by design.
RES_DEG  = 0.01
lon_min  = float(max(np.nanmin(lon_sst), np.nanmin(lon_chl)))
lon_max  = float(min(np.nanmax(lon_sst), np.nanmax(lon_chl)))
lat_min  = float(max(np.nanmin(lat_sst), np.nanmin(lat_chl)))
lat_max  = float(min(np.nanmax(lat_sst), np.nanmax(lat_chl)))

if lon_min >= lon_max or lat_min >= lat_max:
    raise ValueError(
        "SST and Chl-a swaths do not overlap geographically. "
        "Check that both files are from the same satellite pass."
    )

nx = int(round((lon_max - lon_min) / RES_DEG))
ny = int(round((lat_max - lat_min) / RES_DEG))
print(f"Common grid: {nx} × {ny} cells  "
      f"({lon_min:.2f}°–{lon_max:.2f}°E, {lat_min:.2f}°–{lat_max:.2f}°N)")

area_def = geometry.AreaDefinition(
    'common_grid',
    f'Regular {RES_DEG}° grid covering SST/Chl-a overlap',
    'longlat',
    {'proj': 'longlat', 'datum': 'WGS84'},
    nx, ny,
    (lon_min, lat_min, lon_max, lat_max),  # (x_ll, y_ll, x_ur, y_ur) in degrees
)

# Step 3 — resample both swaths onto the common grid.
#   radius_of_influence: maximum distance (metres) to search for a swath
#   neighbour for each output grid cell.  1 500 m is generous for a 0.01°
#   (≈ 1 km) grid while being tight enough to avoid cross-front smearing.
RADIUS_M = 1500
print("Resampling SST onto common grid …")
sst_grid = kd_tree.resample_nearest(
    sst_swath, sst_data, area_def,
    radius_of_influence=RADIUS_M, fill_value=np.nan, nprocs=1,
)
print("Resampling Chl-a onto common grid …")
chl_grid = kd_tree.resample_nearest(
    chl_swath, chl_data, area_def,
    radius_of_influence=RADIUS_M, fill_value=np.nan, nprocs=1,
)

# Step 4 — retain only cells where BOTH products have valid data.
valid = np.isfinite(sst_grid) & np.isfinite(chl_grid) & (chl_grid > 0)
SST_v = sst_grid[valid]
CHL_v = chl_grid[valid]
print(f"Co-located valid pixels: {valid.sum():,}")

if valid.sum() == 0:
    raise RuntimeError(
        "No co-located valid pixels found. "
        "Verify that the SST and Chl-a files are from the same satellite pass "
        "and that both retrieval scripts produced valid output."
    )

# ---- Hexbin scatter (handles large pixel counts efficiently) ----
fig, ax = plt.subplots(figsize=(8, 6))
hb = ax.hexbin(SST_v, np.log10(CHL_v),
               gridsize=80, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=ax, label='Pixel count')

ax.set_xlabel('Sea Surface Temperature (°C)', fontsize=12)
ax.set_ylabel('log$_{10}$(Chlorophyll-a)  [mg/m³]', fontsize=12)
ax.set_title('SST vs Chlorophyll-a — Sentinel-3\n'
             '(cold/upwelling water tends to have higher Chl-a)', fontsize=11)
ax.axhline(0, color='steelblue', linewidth=1, linestyle='--',
           label='1 mg/m³ threshold')
ax.axhline(-1, color='gray', linewidth=0.8, linestyle=':',
           label='0.1 mg/m³')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('SST_ChlorA_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: SST_ChlorA_scatter.png")
