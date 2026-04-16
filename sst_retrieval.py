# =============================================================================
# sst_retrieval.py
# =============================================================================
# PURPOSE:
#   Derive Sea Surface Temperature (SST) from Sentinel-3 SLSTR L1B data
#   using the split-window algorithm, then save a CF-compliant L2 NetCDF
#   and produce a cartopy map.
#
# PHYSICAL BACKGROUND:
#   The ocean surface emits thermal IR radiation according to Planck's law.
#   SLSTR measures this in two channels:
#     S8 (10.85 µm) — "11 µm window channel"
#     S9 (12.0 µm)  — "12 µm window channel"
#   Water vapour in the atmosphere absorbs differently at these two wavelengths.
#   The split-window algorithm exploits the DIFFERENCE (BT11 − BT12) as a
#   proxy for atmospheric water vapour, allowing us to correct for it with
#   simple linear regression coefficients (a0–a3).
#
# KEY LESSON LEARNED DURING TESTING:
#   SLSTR stores BTs as scaled 16-bit integers to save disk space.
#   Physical value = raw_integer * scale_factor + add_offset.
#   The fix is xr.open_dataset(..., mask_and_scale=True), which applies
#   the scale/offset automatically and replaces _FillValue with NaN.
#   Without this, raw integers (~10000) look like physically impossible
#   temperatures (-58°C), which caused 0.1% valid pixels in early tests.
#
# ALGORITHM REFERENCE:
#   Merchant, C.J. et al. (2019). Satellite-based time-series of sea-surface
#   temperature since 1981 for climate applications. Scientific Data, 6, 223.
#   doi:10.1038/s41597-019-0236-x
#
# PREREQUISITE:
#   SL_1_RBT zip file in the working directory (from download_sentinel3.py)
#
# OUTPUTS:
#   sentinel3_SST_L2.nc   -- L2 SST in °C with CF-convention attributes
#   SST_L2_map.png        -- Publication-quality cartopy map
# =============================================================================
# sst_retrieval.py
import glob
import os
import zipfile
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean


def unzip_sentinel(product_type, extract_dir='.'):
    """Auto-detect and unzip a Sentinel-3 product by type."""
    matches = glob.glob(f'S3*{product_type}*.zip')
    if len(matches) == 0:
        raise FileNotFoundError(f"No Sentinel-3 {product_type} zip file found in {os.getcwd()}")
    if len(matches) > 1:
        print("Multiple zip files found, using the first:")
        for m in matches: print(f"  {m}")
    zip_path = matches[0]
    print(f"Found: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    sen3_path = zip_path.replace('.SEN3.zip', '.SEN3').replace('.zip', '.SEN3')
    if not os.path.exists(sen3_path):
        raise FileNotFoundError(f"Expected .SEN3 folder not found: {sen3_path}")
    print(f"Unzipped: {sen3_path}")
    return sen3_path + '/'


# ---- Unzip SLSTR L1B ----
slstr_path = unzip_sentinel('SL_1_RBT')

# ---- Step 1: Load brightness temperatures ----
# CRITICAL: mask_and_scale=True is essential for SLSTR data.
# Without it, xarray returns raw 16-bit integers, NOT physical temperatures.
# With it, xarray automatically applies:
#   physical_value = raw * scale_factor + add_offset
# and replaces the _FillValue sentinel with NaN.
# This is the fix for the -58°C / 0.1% valid pixels bug seen in testing.
ds_s8 = xr.open_dataset(slstr_path + 'S8_BT_in.nc', mask_and_scale=True)
ds_s9 = xr.open_dataset(slstr_path + 'S9_BT_in.nc', mask_and_scale=True)

BT11 = ds_s8['S8_BT_in'].values.astype(float)  # Kelvin after auto-scaling
BT12 = ds_s9['S9_BT_in'].values.astype(float)

print(f"BT11 raw range (K): {np.nanmin(BT11):.1f} – {np.nanmax(BT11):.1f}")
print(f"BT12 raw range (K): {np.nanmin(BT12):.1f} – {np.nanmax(BT12):.1f}")

# ---- Step 2: Load geolocation ----
ds_geo = xr.open_dataset(slstr_path + 'geodetic_in.nc', mask_and_scale=True)
lat = ds_geo['latitude_in'].values
lon = ds_geo['longitude_in'].values

# ---- Step 3: Load cloud flags ----
ds_flag = xr.open_dataset(slstr_path + 'flags_in.nc')
cloud_flag = ds_flag['cloud_in'].values

# ---- Step 4: Load satellite zenith angle ----
ds_geom = xr.open_dataset(slstr_path + 'geometry_tn.nc', mask_and_scale=True)
sza_mean = np.nanmean(ds_geom['sat_zenith_tn'].values)
sec_theta = 1.0 / np.cos(np.radians(sza_mean))

# ---- Step 5: Masking — two-stage approach ----
# Stage 1 — Physical range filter:
#   Ocean surface temperatures span roughly 270–320 K (−3°C to +47°C).
#   Using a physical range rather than the product's _FillValue is more
#   robust because it catches any residual unphysical values regardless of
#   how the fill value was encoded.
#
# Stage 2 — Cloud masking:
#   The cloud_in bitmask uses individual bits to flag different cloud tests.
#   Bit 0 (value=1) is the "gross cloud" flag — the most conservative.
#   We use a bitwise AND (&) to isolate that bit:
#     (cloud_flag & 1) == 1  means bit 0 is set → cloud → NaN
# Ocean BTs should be roughly 270–310 K; anything outside is invalid
BT11 = np.where((BT11 > 270) & (BT11 < 320), BT11, np.nan)
BT12 = np.where((BT12 > 270) & (BT12 < 320), BT12, np.nan)

# Mask cloud pixels (bit 0 of cloud_in = gross cloud)
cloud_mask = (cloud_flag & 1).astype(bool)
BT11[cloud_mask] = np.nan
BT12[cloud_mask] = np.nan

# ---- Step 6: Split-window SST algorithm ----
# The algorithm has three additive terms:
#   a0 + a1*BT11          — base SST from the 11 µm channel alone
#   a2*(BT11-BT12)        — water vapour correction (the "split-window")
#   a3*(BT11-BT12)*(secθ-1) — scan-angle correction for longer path lengths
#                              through the atmosphere at oblique viewing
#
# sec(θ) = 1/cos(θ): at nadir (θ=0°) sec=1 so the third term vanishes.
# At larger zenith angles the atmosphere correction becomes more important.
# We use the scene-mean zenith angle — a simplification adequate for class.
a0 = -2.6836
a1 =  1.0029
a2 =  0.8641
a3 =  0.6209

SST_K = a0 + a1 * BT11 + a2 * (BT11 - BT12) + a3 * (BT11 - BT12) * (sec_theta - 1)
SST_celsius = SST_K - 273.15

# ---- Step 7: Sanity check ----
print(f"SST range: {np.nanmin(SST_celsius):.1f} to {np.nanmax(SST_celsius):.1f} °C")
valid_fraction = np.sum(np.isfinite(SST_celsius)) / SST_celsius.size * 100
print(f"Valid pixels: {valid_fraction:.1f}%")

if valid_fraction < 1.0:
    print("\nWARNING: Very few valid pixels. Printing BT11 percentiles for diagnosis:")
    raw = ds_s8['S8_BT_in'].values.astype(float)
    print("  BT11 percentiles (2, 25, 50, 75, 98):",
          np.nanpercentile(raw, [2, 25, 50, 75, 98]))

# ---- Step 8: Save as L2 NetCDF (CF-convention) ----
# CF (Climate and Forecast) conventions are the standard metadata format
# for geoscience NetCDF files. Key CF elements we include:
#   long_name   — human-readable description of the variable
#   units       — physical units (UDUNITS-compatible string)
#   processing_level — "L2" per the satellite data level convention
# The lat/lon are stored as non-dimension coordinates on the 2-D (y,x) grid,
# preserving the swath geometry. This is standard for L2 swath products.
sst_da = xr.DataArray(
    SST_celsius,
    name='sst',
    dims=['y', 'x'],
    coords={'lat': (['y', 'x'], lat), 'lon': (['y', 'x'], lon)},
    attrs={
        'long_name': 'Sea Surface Temperature',
        'units': 'degrees_Celsius',
        'processing_level': 'L2',
        'source': 'Sentinel-3 SLSTR L1B, split-window algorithm'
    }
)
sst_da.to_netcdf('sentinel3_SST_L2.nc')
print('L2 SST saved to sentinel3_SST_L2.nc')

# ---- Step 9: Visualise with cartopy ----
# cartopy handles the geographic projection and coastline/border overlays.
# Key choices:
#   PlateCarree — simple equirectangular projection, good for regional maps
#   pcolormesh  — correct for irregular grids (each pixel at its own lat/lon)
#                 DO NOT use imshow() on swath data — it ignores coordinates
#   vmin/vmax from 2nd–98th percentiles — robust to outlier pixels; avoids
#     the colourscale being dominated by a few extreme cloud-edge values
#   cmocean.cm.thermal — perceptually uniform, colourblind-safe colourmap
#     specifically designed for SST; progresses dark→light with increasing T
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

valid = np.isfinite(SST_celsius)
ax.set_extent([
    np.nanmin(lon[valid]), np.nanmax(lon[valid]),
    np.nanmin(lat[valid]), np.nanmax(lat[valid])
], crs=ccrs.PlateCarree())

img = ax.pcolormesh(
    lon, lat, SST_celsius,
    cmap=cmocean.cm.thermal,
    vmin=np.nanpercentile(SST_celsius[valid], 2),
    vmax=np.nanpercentile(SST_celsius[valid], 98),
    transform=ccrs.PlateCarree(),
    rasterized=True
)
plt.colorbar(img, ax=ax, label='SST (°C)', pad=0.02, shrink=0.8)

ax.add_feature(cfeature.LAND, color='lightgray', zorder=3)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=4)
ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':', zorder=4)
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
ax.set_title('Sea Surface Temperature (SST) — Sentinel-3 SLSTR L2\n'
             '(Split-window algorithm from L1B BTs)', fontsize=12, pad=10)

plt.tight_layout()
plt.savefig('SST_L2_map.png', dpi=200, bbox_inches='tight')
plt.show()
