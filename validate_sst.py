# =============================================================================
# validate_sst.py
# =============================================================================
# PURPOSE:
#   Compare our split-window SST retrieval (from sst_retrieval.py) against
#   the official EUMETSAT SLSTR L2 WST (Water Surface Temperature) product
#   for the same satellite granule. Computes standard validation statistics
#   and produces a three-panel diagnostic figure.
#
# WHY VALIDATE?
#   Satellite-derived geophysical products always need validation against a
#   reference — either in-situ measurements (moored buoys, Argo floats, ship
#   surveys) or an official operational product. Validation:
#   • Quantifies the accuracy of our simplified algorithm
#   • Reveals systematic biases (e.g. our SST is consistently 0.5°C too warm)
#   • Identifies spatial patterns in the errors (e.g. more bias near coasts)
#   • Builds confidence before using the data for scientific analysis
#
# THE REFERENCE PRODUCT (SL_2_WST___):
#   EUMETSAT's operational SST product, produced with the full GHRSST L2P
#   processing chain. It uses the same split-window algorithm but with
#   spatially and temporally varying coefficients calibrated against the
#   global iQuam in-situ database. It also applies a more sophisticated
#   cloud mask. Bias vs iQuam: typically < 0.3°C RMSE over open ocean.
#
# VALIDATION METRICS:
#   Bias = mean(our - ref)          → systematic offset (sign matters)
#   RMSE = sqrt(mean((our-ref)^2))  → overall spread including bias
#   R    = Pearson correlation       → how well patterns are reproduced
#   Target performance for a teaching exercise: |Bias| < 1°C, RMSE < 1.5°C
#   Operational EUMETSAT target: RMSE < 0.5°C (GHRsst L4 requirement)
#
# CO-REGISTRATION (crucial step):
#   Our SST and the reference WST are on different irregular swath grids.
#   Direct array comparison would fail (shape mismatch) or give nonsense
#   (comparing unrelated pixels). We use pyresample to resample the reference
#   onto our swath grid so each pixel pair is geographically co-located.
#
# THREE-PANEL FIGURE:
#   Panel A (scatter): hexbin of our vs reference — ideal = points on 1:1 line
#   Panel B (diff map): spatial pattern of our-ref — reveals geographic biases
#   Panel C (reference map): what the reference looks like — sanity check
#
# BUGS FIXED (compared to original course document version):
#   1. __xarray_dataarray_variable__ key replaced by data_vars[0]
#   2. Hardcoded <datetime> path replaced by glob + unzip_sentinel()
#   3. No co-registration → fixed with pyresample
#   4. plt.show() → plt.close() (avoids hanging in non-interactive mode)
#
# PREREQUISITE:
#   1. Run sst_retrieval.py first → sentinel3_SST_L2.nc
#   2. Download SL_2_WST___ for the SAME orbit via download_sentinel3.py
# =============================================================================
# validate_sst.py
# Compares our derived SST (from sst_retrieval.py) against the official
# EUMETSAT SLSTR L2 WST product (SL_2_WST) for the same granule.
#
# PREREQUISITES:
#   1. Run sst_retrieval.py first  ->  sentinel3_SST_L2.nc
#   2. Download the matching SL_2_WST product via download_sentinel3.py
#      (add productType='SL_2_WST___' to the download script)
#      OR use the reproject outputs: sentinel3_SST_grid.nc as a self-check.
#
# WHAT THIS SCRIPT DOES:
#   - Loads both SST fields
#   - Regrids the reference onto the same swath geometry as our retrieval
#     (using pyresample nearest-neighbour, same approach as reproject.py)
#   - Computes Bias, RMSE and Pearson R
#   - Produces a hexbin scatter plot and a difference map

import glob
import os
import zipfile
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from pyresample import geometry, kd_tree


# -----------------------------------------------------------------------
# 1. Load our derived L2 SST (output of sst_retrieval.py)
# -----------------------------------------------------------------------
def find_file(pattern, description):
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No {description} found matching '{pattern}'.\n"
            f"Run the required script first.")
    print(f"Using {description}: {matches[0]}")
    return matches[0]

our_file = find_file('sentinel3_SST_L2*.nc', 'our SST L2')
ds_our   = xr.open_dataset(our_file)
varname  = [v for v in ds_our.data_vars][0]
our_sst  = ds_our[varname].values.astype(np.float32)   # (rows, cols) in °C
lat_our  = ds_our['lat'].values.astype(np.float32)
lon_our  = ds_our['lon'].values.astype(np.float32)
ds_our.close()
print(f"Our SST shape: {our_sst.shape}  range: "
      f"{np.nanmin(our_sst):.1f} – {np.nanmax(our_sst):.1f} °C")


# -----------------------------------------------------------------------
# 2. Load reference EUMETSAT L2 WST (SL_2_WST product)
# -----------------------------------------------------------------------
def unzip_sentinel(product_type, extract_dir='.'):
    """Auto-detect and unzip a Sentinel-3 product by type."""
    matches = glob.glob(f'S3*{product_type}*.zip')
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No SL_2_WST zip file found in {os.getcwd()}.\n"
            f"Download it with download_sentinel3.py using "
            f"productType='SL_2_WST___' for the same date/orbit as your L1B.")
    zip_path = matches[0]
    sen3_path = zip_path.replace('.SEN3.zip', '.SEN3').replace('.zip', '.SEN3')
    if not os.path.exists(sen3_path):
        print(f"Unzipping {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    else:
        print(f"Already unzipped: {sen3_path}")
    return sen3_path + '/'

wst_path = unzip_sentinel('SL_2_WST')

# The WST product stores SST in 'sea_surface_temperature' (Kelvin).
# Older multi-file layout: SST_nt.nc (night) or SST_dt.nc (day).
# Newer GHRSST L2P layout: a single *SSTskin*.nc file.
sst_file = None
for candidate in ['SST_nt.nc', 'SST_dt.nc']:
    p = wst_path + candidate
    if os.path.exists(p):
        sst_file = p
        break

if sst_file is None:
    # GHRSST L2P layout: one *.nc file (not the manifest) in the .SEN3 folder
    nc_files = [f for f in glob.glob(wst_path + '*.nc')
                if 'xfdu' not in f and 'manifest' not in f]
    if nc_files:
        sst_file = nc_files[0]

if sst_file is None:
    contents = os.listdir(wst_path)
    raise FileNotFoundError(
        f"Could not find SST_nt.nc, SST_dt.nc, or any *.nc in {wst_path}.\n"
        f"Contents: {contents}")

print(f"Reference SST file: {sst_file}")
ds_ref   = xr.open_dataset(sst_file, mask_and_scale=True)
ref_sst_K = ds_ref['sea_surface_temperature'].values.astype(np.float32)
ref_sst   = ref_sst_K - 273.15    # Kelvin -> Celsius

# WST geolocation is in a separate geodetic file
geo_file = wst_path + 'geodetic_in.nc'
if not os.path.exists(geo_file):
    # Some versions store it directly in the SST file
    lat_ref = ds_ref['lat'].values.astype(np.float32)
    lon_ref = ds_ref['lon'].values.astype(np.float32)
else:
    ds_geo_ref = xr.open_dataset(geo_file, mask_and_scale=True)
    lat_ref = ds_geo_ref['latitude_in'].values.astype(np.float32)
    lon_ref = ds_geo_ref['longitude_in'].values.astype(np.float32)
    ds_geo_ref.close()
ds_ref.close()

print(f"Reference SST shape: {ref_sst.shape}  range: "
      f"{np.nanmin(ref_sst):.1f} – {np.nanmax(ref_sst):.1f} °C")


# -----------------------------------------------------------------------
# 3. Co-register: resample reference onto our swath grid
# -----------------------------------------------------------------------
# WHY RESAMPLE THE REFERENCE ONTO OUR GRID (not the other way around)?
#   Our derived SST is the "test" product — it has whatever shape it has
#   (possibly subsampled). The reference is the "truth" baseline. We warp
#   the reference to match our grid so that:
#   • No resampling is applied to our data (we validate it as-is)
#   • Both arrays end up with the same shape → direct subtraction works
#   • The output arrays are in the same coordinate system as our maps
#
# SwathDefinition vs SwathDefinition resampling:
#   Unlike reproject.py where we resample to an AreaDefinition (regular grid),
#   here we resample from one irregular swath to another irregular swath.
#   This is less common but fully supported by pyresample — the target is
#   defined by our lat/lon arrays rather than a regular grid description.
# Our SST may be subsampled (SUBSAMPLE=2) relative to the reference 1 km grid.
# We resample the reference onto our lat/lon grid using pyresample so every
# pixel is directly comparable.
print("Co-registering reference SST onto our swath grid...")
swath_ref = geometry.SwathDefinition(lons=lon_ref, lats=lat_ref)
swath_our = geometry.SwathDefinition(lons=lon_our, lats=lat_our)

ref_on_our = kd_tree.resample_nearest(
    swath_ref, ref_sst,
    swath_our,
    radius_of_influence=3000,   # metres
    fill_value=np.nan,
    epsilon=0.5,
).astype(np.float32)

print(f"Co-registered shape: {ref_on_our.shape}")


# -----------------------------------------------------------------------
# 4. Validation statistics (over co-located valid pixels)
# -----------------------------------------------------------------------
# We only compute statistics where BOTH our SST and the reference are valid
# (non-NaN). Pixels masked as cloud in either product are excluded.
# This is the standard "matchup" approach in satellite validation.
#
# Bias interpretation:
#   Positive bias: our SST > reference (we overestimate temperature)
#   Negative bias: our SST < reference (we underestimate temperature)
#   Common causes of bias: imperfect cloud mask, simplified algorithm,
#   skin vs bulk temperature differences, thin cirrus not caught by flags.
#
# WARNING about low matchup counts:
#   If valid.sum() < 100, the products are probably from different orbits
#   or dates. Check that the L1B granule and WST granule have the same
#   acquisition time and orbit number in their filenames.
valid = np.isfinite(our_sst) & np.isfinite(ref_on_our)
our_v = our_sst[valid]
ref_v = ref_on_our[valid]

print(f"\nCo-located valid pixels: {valid.sum():,}")
if valid.sum() < 100:
    print("WARNING: very few co-located pixels — check that the L1B and WST "
          "products are from the same orbit/granule.")

bias = np.mean(our_v - ref_v)
rmse = np.sqrt(np.mean((our_v - ref_v)**2))
r    = np.corrcoef(our_v, ref_v)[0, 1]
print(f"Bias : {bias:+.3f} °C")
print(f"RMSE : {rmse:.3f} °C")
print(f"R    : {r:.4f}")


# -----------------------------------------------------------------------
# 5. Figures
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(16, 6))

# --- Panel A: scatter ---
ax1 = fig.add_subplot(1, 3, 1)
hb  = ax1.hexbin(ref_v, our_v, gridsize=80, cmap='Blues', mincnt=1)
plt.colorbar(hb, ax=ax1, label='Pixel count')
lims = [min(ref_v.min(), our_v.min()) - 0.5,
        max(ref_v.max(), our_v.max()) + 0.5]
ax1.plot(lims, lims, 'r--', linewidth=1, label='1:1 line')
ax1.set_xlim(lims); ax1.set_ylim(lims)
ax1.set_xlabel('Reference SST — EUMETSAT L2 WST (°C)')
ax1.set_ylabel('Our derived SST (°C)')
ax1.set_title(f'SST Validation\nBias={bias:+.3f}°C  RMSE={rmse:.3f}°C  R={r:.4f}',
              fontsize=10)
ax1.legend(fontsize=9)

# --- Panel B: difference map ---
diff = our_sst - ref_on_our
diff_lim = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95)

ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
valid_pts = np.isfinite(our_sst)
ax2.set_extent([np.nanmin(lon_our[valid_pts]), np.nanmax(lon_our[valid_pts]),
                np.nanmin(lat_our[valid_pts]), np.nanmax(lat_our[valid_pts])],
               crs=ccrs.PlateCarree())
im2 = ax2.pcolormesh(lon_our, lat_our, diff,
                     cmap='RdBu_r',
                     vmin=-diff_lim, vmax=diff_lim,
                     transform=ccrs.PlateCarree(), rasterized=True)
plt.colorbar(im2, ax=ax2, label='Our SST − Reference (°C)', shrink=0.85)
ax2.add_feature(cfeature.LAND, color='#d4c5a9', zorder=3)
ax2.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=4)
ax2.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5)
ax2.set_title('Difference map\n(Our − EUMETSAT WST)', fontsize=10)

# --- Panel C: reference SST map ---
ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
ax3.set_extent([np.nanmin(lon_our[valid_pts]), np.nanmax(lon_our[valid_pts]),
                np.nanmin(lat_our[valid_pts]), np.nanmax(lat_our[valid_pts])],
               crs=ccrs.PlateCarree())
im3 = ax3.pcolormesh(lon_our, lat_our, ref_on_our,
                     cmap=cmocean.cm.thermal,
                     vmin=np.nanpercentile(ref_v, 2),
                     vmax=np.nanpercentile(ref_v, 98),
                     transform=ccrs.PlateCarree(), rasterized=True)
plt.colorbar(im3, ax=ax3, label='Reference SST (°C)', shrink=0.85)
ax3.add_feature(cfeature.LAND, color='#d4c5a9', zorder=3)
ax3.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=4)
ax3.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5)
ax3.set_title('Reference: EUMETSAT L2 WST', fontsize=10)

fig.suptitle('SST Validation — Sentinel-3 SLSTR\n'
             '(Our split-window retrieval vs official EUMETSAT L2 WST)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('SST_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: SST_validation.png")
