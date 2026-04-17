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
# PREREQUISITE:
#   Run sst_retrieval.py and chlora_retrieval.py first.
# =============================================================================
# sst_chla_scatter.py
# SST vs Chlor-a scatter plot — explores the biological relationship
# between thermal stratification and phytoplankton biomass

import glob
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


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


def load_l2_flat(filepath, varname):
    """Open a L2 NetCDF and return flattened 1-D data array."""
    ds = xr.open_dataset(filepath)
    data = ds[varname].values.astype(float).ravel()
    ds.close()
    return data


# ---- Auto-detect L2 files ----
sst_file = find_l2_netcdf('sentinel3_SST_L2*.nc',   'SST L2')
chl_file = find_l2_netcdf('sentinel3_ChlorA_L2*.nc', 'Chlor-a L2')

SST = load_l2_flat(sst_file, 'sst')
CHL = load_l2_flat(chl_file, 'chla')

# ---- Co-location of SST and Chlor-a ----
# SST (SLSTR, ~1 km native resolution) and Chlor-a (OLCI, 300 m native,
# but subsampled to ~600 m by chlora_retrieval.py) have DIFFERENT grid sizes.
# After flattening to 1-D with .ravel(), they will have different lengths.
#
# SIMPLE APPROACH USED HERE (adequate for a teaching exercise):
#   Trim both arrays to the shorter length with [:min_len].
#   This implicitly assumes a "nearest-pixel" co-location from the top-left
#   of the scene outward. It is NOT geometrically correct (pixels from
#   different positions are compared), but the bias is small for a scatter
#   plot exploring a general ecological relationship.
#
# RIGOROUS APPROACH (see reproject.py):
#   Resample both products onto a common regular grid using pyresample,
#   then compare pixel-by-pixel. This is required for quantitative analysis.
min_len = min(len(SST), len(CHL))
SST = SST[:min_len]
CHL = CHL[:min_len]

valid = np.isfinite(SST) & np.isfinite(CHL)
SST_v = SST[valid]
CHL_v = CHL[valid]
print(f"Co-located valid pixels: {valid.sum():,}")

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
