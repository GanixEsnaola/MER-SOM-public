# =============================================================================
# true_color_map.py
# =============================================================================
# PURPOSE:
#   Produce a quasi-true colour RGB composite from Sentinel-3 OLCI L1B.
#   Uses the same OL_1_EFR product as chlora_retrieval.py — no extra download.
#
# BAND-TO-CHANNEL MAPPING:
#   Oa08  665 nm  →  Red channel    (near the peak of human red sensitivity)
#   Oa06  560 nm  →  Green channel  (green light, strong phytoplankton signal)
#   Oa03  443 nm  →  Blue channel   (blue light, absorbed by chlorophyll)
#   This is called "quasi-true colour" because OLCI band wavelengths do not
#   exactly match the human eye's RGB sensitivities (700/546/436 nm). The
#   result looks approximately natural but colours are not photographic.
#
# GAMMA CORRECTION — WHY IT IS NEEDED:
#   Ocean water-leaving reflectances are very low: ρ_w ≈ 0.01–0.05 (1–5%).
#   Without gamma correction, the whole image appears nearly black.
#   Gamma correction: output = input^gamma
#   For gamma < 1: dark values are stretched upward, bright values compressed.
#   At gamma=0.5 (square root): ρ_w=0.01 → 0.10, ρ_w=0.04 → 0.20.
#   NASA Worldview and EUMETSAT EUMETview use gamma ≈ 0.45–0.5 for ocean RGB.
#   Try gamma=0.4 for darker (more natural) results, 0.6 for brighter.
#
# WHAT TO LOOK FOR IN THE RGB IMAGE:
#   Dark blue/navy   — clear oligotrophic open ocean (low particles)
#   Turquoise/cyan   — dense phytoplankton bloom (high backscatter)
#   Milky white-blue — coccolithophore bloom (CaCO3 plates scatter strongly)
#   Brown/yellow     — river plume or coastal sediment (CDOM + SPM)
#   White            — cloud or foam (masked to white by our script)
#   The RGB image and the Chlor-a map together are very powerful: you can
#   see the spatial structure of blooms visually and then quantify biomass.
#
# MEMORY NOTE:
#   We skip re-unzipping if the .SEN3 folder already exists (from
#   chlora_retrieval.py) to avoid redundant disk I/O. The same SUBSAMPLE
#   and float32 strategy is used for memory efficiency.
#
# PREREQUISITE:
#   OL_1_EFR zip file (or already-unzipped .SEN3 folder) in working directory.
# =============================================================================
# true_color_map.py
# Produces a quasi-true colour RGB composite from Sentinel-3 OLCI L1B.
# Uses bands: Oa08 (665 nm = Red), Oa06 (560 nm = Green), Oa03 (443 nm = Blue).
# No extra download needed — works from the same OL_1_EFR product as chlora_retrieval.py.
#
# Processing steps:
#   1. Load radiances (mask_and_scale=True handles fill/scale automatically)
#   2. Convert to TOA reflectance: rho = pi * L / (E0 * cos(SZA))
#   3. Subtract Rayleigh contribution (same simplified model as chlora_retrieval.py)
#   4. Normalise to [0, 1] using scene percentiles
#   5. Apply gamma correction to brighten (gamma < 1 brightens)
#   6. Stack R/G/B and plot with cartopy

import glob
import os
import zipfile
import xarray as xr
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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
    # Skip unzipping if already extracted
    sen3_path = zip_path.replace('.SEN3.zip', '.SEN3').replace('.zip', '.SEN3')
    if os.path.exists(sen3_path):
        print(f"Already unzipped: {sen3_path}")
    else:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        print(f"Unzipped: {sen3_path}")
    return sen3_path + '/'


# ---- Configuration ----
SUBSAMPLE = 2    # Match the subsampling used in chlora_retrieval.py
GAMMA     = 0.5  # Gamma < 1 brightens the image (try 0.4–0.6 for ocean scenes)

# RGB band definitions: (band_name, wavelength_nm, OLCI_band_index_0based)
RGB_BANDS = [
    ('Oa08', 665, 7),   # Red
    ('Oa06', 560, 5),   # Green
    ('Oa03', 443, 2),   # Blue
]

# ---- Unzip OLCI L1B (skips if already done by chlora_retrieval.py) ----
olci_path = unzip_sentinel('OL_1_EFR')

# ---- Solar flux (netCDF4 to avoid xarray duplicate-dimension warning) ----
with nc.Dataset(olci_path + 'instrument_data.nc') as ds_nc:
    solar_flux_raw = ds_nc.variables['solar_flux'][:]
if solar_flux_raw.ndim == 2:
    E0_all = np.nanmean(solar_flux_raw, axis=0)
    if E0_all.shape[0] != 21:
        E0_all = np.nanmean(solar_flux_raw, axis=1)
else:
    E0_all = solar_flux_raw

# ---- Solar zenith angle (tie-point grid) ----
ds_tie = xr.open_dataset(olci_path + 'tie_geometries.nc', mask_and_scale=True)
sza_tie = ds_tie['SZA'].values.astype(np.float32)
ds_tie.close()

# ---- Geolocation (subsampled) ----
ds_geo = xr.open_dataset(olci_path + 'geo_coordinates.nc', mask_and_scale=True)
lat = ds_geo['latitude'].values[::SUBSAMPLE, ::SUBSAMPLE].astype(np.float32)
lon = ds_geo['longitude'].values[::SUBSAMPLE, ::SUBSAMPLE].astype(np.float32)
ds_geo.close()

# ---- Quality flags (subsampled) ----
ds_qf = xr.open_dataset(olci_path + 'qualityFlags.nc')
qf = ds_qf['quality_flags'].values[::SUBSAMPLE, ::SUBSAMPLE]
ds_qf.close()
invalid_mask = ((qf & (1 << 31)) != 0) | ((qf & (1 << 27)) != 0) | ((qf & (1 << 25)) != 0)
del qf

# ---- Upsample SZA to subsampled pixel grid ----
full_rows, full_cols = lat.shape
tie_rows,  tie_cols  = sza_tie.shape
row_idx  = np.round(np.linspace(0, tie_rows - 1, full_rows)).astype(int)
col_idx  = np.round(np.linspace(0, tie_cols - 1, full_cols)).astype(int)
sza_full = sza_tie[np.ix_(row_idx, col_idx)]
cos_sza  = np.cos(np.radians(sza_full))
del sza_tie

# ---- Process each RGB band: radiance -> TOA reflectance -> Rayleigh-corrected ----
rgb_channels = {}
for band_name, wl, band_idx in RGB_BANDS:
    # Load radiance
    ds = xr.open_dataset(olci_path + f'{band_name}_radiance.nc', mask_and_scale=True)
    rad = ds[f'{band_name}_radiance'].values.astype(np.float32)[::SUBSAMPLE, ::SUBSAMPLE]
    ds.close()

    E0 = float(E0_all[band_idx])

    # TOA reflectance
    rho_toa = (np.pi * rad) / (E0 * cos_sza)

    # Simplified Rayleigh correction (same model as chlora_retrieval.py)
    lam   = wl / 1000.0
    tau_r = 0.008569 * lam**-4 * (1 + 0.0113 * lam**-2)
    rho_r = 0.75 * tau_r
    rho_w = rho_toa - rho_r

    # Mask invalid pixels
    rho_w[invalid_mask] = np.nan
    rho_w[rho_w < 0]    = 0.0   # clip negatives to 0 (don't NaN — keeps spatial context)

    rgb_channels[band_name] = rho_w
    print(f"  {band_name} ({wl} nm): rho_w range "
          f"{np.nanmin(rho_w):.4f} – {np.nanmax(rho_w):.4f}")

del cos_sza, invalid_mask

# ---- Normalise each channel to [0, 1] using robust percentiles ----
# Each RGB channel must be in [0,1] for display. We use percentile clipping
# rather than true min/max for robustness:
#   • A few anomalously bright pixels (cloud edges, ship glint) would cause
#     the true maximum to be very large, compressing the ocean into a tiny
#     fraction of the colour range.
#   • The 1st–99th percentile clip ignores the top and bottom 1% of values,
#     so the colour range captures the typical ocean scene variation.
# Each channel is normalised INDEPENDENTLY — this adjusts for the different
# absolute reflectance levels at 443/560/665 nm, which is why the image
# does not look spectrally accurate but does look visually natural.
def normalise(arr, pmin=1, pmax=99):
    """Stretch array to [0,1] using percentile clipping."""
    valid = arr[np.isfinite(arr) & (arr > 0)]
    lo = np.percentile(valid, pmin)
    hi = np.percentile(valid, pmax)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1)

R = normalise(rgb_channels['Oa08'])
G = normalise(rgb_channels['Oa06'])
B = normalise(rgb_channels['Oa03'])
del rgb_channels

# ---- Gamma correction: brighten the image ----
# Ocean reflectances are very low (~0.01–0.05); gamma < 1 stretches dark values
R = np.power(R, GAMMA)
G = np.power(G, GAMMA)
B = np.power(B, GAMMA)

# ---- Stack into RGB array (H × W × 3), NaN → white background ----
# np.dstack stacks three (H,W) arrays along a new third axis → shape (H,W,3).
# imshow() interprets the third axis as R, G, B colour channels.
#
# NaN pixels (land, cloud, quality-flagged) are set to white (1,1,1).
# We choose white rather than black because:
#   - Black (0,0,0) is also a valid very-dark ocean colour
#   - White is unambiguous and visually distinct from any ocean signature
#   - Cartopy's LAND feature covers land pixels anyway (zorder=4)
rgb = np.dstack([R, G, B]).astype(np.float32)
nan_mask = ~np.isfinite(rgb).all(axis=2)
rgb[nan_mask] = 1.0   # white for masked pixels (land, cloud, invalid)
del R, G, B

# ---- Plot with cartopy ----
fig = plt.figure(figsize=(12, 8))
ax  = plt.axes(projection=ccrs.PlateCarree())

# IMPORTANT NOTE ON imshow() vs pcolormesh() for swath data:
# pcolormesh() accepts individual lat/lon per pixel → correct for irregular grids.
# imshow() assumes the image is already on a regular grid, but is the only
# matplotlib function that can display a 3-channel (H×W×3) RGB array.
# Our compromise: use imshow() with the overall lat/lon bounding box as extent.
# This introduces a small distortion where the swath is not perfectly regular,
# but for the purpose of a quick-look RGB image it is visually acceptable.
# For geometrically precise RGB: reproject to a regular grid first (reproject.py)
# then display with imshow.
#
# CRITICAL: both ax.set_extent AND imshow's extent parameter MUST use the same
# bounding box — the full swath extent (ALL pixels, including white land/invalid
# ones). Using only valid (ocean) pixels for ax.set_extent while imshow uses the
# full extent places the image array in a larger box than the map viewport shows,
# shifting all ocean pixels onto the land portion of the map.
lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)

ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Determine row ordering: if row 0 is the northernmost scan line (descending
# orbit, typical for daytime OLCI), origin='upper' is correct.  For ascending
# passes row 0 is southernmost and origin='lower' is needed.
mid_col = lat.shape[1] // 2
origin = 'upper' if lat[0, mid_col] > lat[-1, mid_col] else 'lower'

ax.imshow(
    rgb,
    origin=origin,
    extent=[lon_min, lon_max, lat_min, lat_max],
    transform=ccrs.PlateCarree(),
    interpolation='none',
    aspect='auto',
)

ax.add_feature(cfeature.LAND,      color='lightgray', zorder=3)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8,    zorder=4, edgecolor='black')
ax.add_feature(cfeature.BORDERS,   linewidth=0.4,    zorder=4, edgecolor='black',
               linestyle=':')
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6)
ax.set_title(
    f'Quasi-True Colour RGB — Sentinel-3 OLCI L1B\n'
    f'R=Oa08 (665 nm)  G=Oa06 (560 nm)  B=Oa03 (443 nm) | gamma={GAMMA}',
    fontsize=11, pad=10
)

plt.tight_layout()
plt.savefig('TrueColor_RGB_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: TrueColor_RGB_map.png")
