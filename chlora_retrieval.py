# =============================================================================
# chlora_retrieval.py
# =============================================================================
# PURPOSE:
#   Derive Chlorophyll-a concentration (Chlor-a, mg/m³) from Sentinel-3
#   OLCI L1B radiances using a simplified Rayleigh atmospheric correction
#   followed by the OC4ME ocean colour algorithm.
#
# WHY IS THIS HARDER THAN SST?
#   For SST we measure thermal emission from the surface — the atmosphere is
#   relatively transparent in the thermal IR. For ocean colour, we measure
#   solar radiation reflected by the water — and ~90% of that signal comes
#   from the atmosphere (Rayleigh + aerosol scattering), not the ocean.
#   Removing the atmospheric contribution is called "atmospheric correction"
#   and is the most challenging step in ocean colour remote sensing.
#
# OUR SIMPLIFIED APPROACH (Rayleigh-only):
#   We subtract only the Rayleigh scattering contribution (molecular
#   scattering by air molecules, well-characterised analytically).
#   We OMIT aerosol correction — this is acceptable for clear open-ocean
#   scenes with low aerosol loading, but will bias Chlor-a high in
#   coastal, hazy, or dusty scenes.
#   For production work: use C2RCC (ESA SNAP) or download the L2 WFR product.
#
# THE OC4ME ALGORITHM:
#   Ocean Colour 4-band Maximum band ratio Empirical algorithm.
#   Exploits the fact that phytoplankton chlorophyll absorbs blue light
#   (~443 nm) and reflects green (~560 nm). The blue/green ratio therefore
#   decreases as Chlor-a increases.
#   A 4th-order polynomial in log10(band_ratio) gives log10(Chl-a).
#   Coefficients are from the IOCCG/ESA standard calibration dataset.
#
# MEMORY MANAGEMENT:
#   A full-resolution OLCI scene is ~4091×4865 pixels = ~20M pixels.
#   At float64, each array = 160 MB. With multiple arrays in memory
#   simultaneously, a small VM (4–8 GB RAM) will be killed by the OS.
#   We use two strategies:
#     1. SUBSAMPLE: read every 2nd pixel → ~5M pixels → 4× less memory
#     2. float32 instead of float64 → half the memory per array
#     3. del after use → allow garbage collection before next step
#
# PREREQUISITE:
#   OL_1_EFR zip file in the working directory (from download_sentinel3.py)
#
# OUTPUTS:
#   sentinel3_ChlorA_L2.nc   -- L2 Chlor-a in mg/m³
#   ChlorA_L2_map.png        -- Map with logarithmic colour scale
# =============================================================================
# chlora_retrieval.py
# Works from OLCI L1B (OL_1_EFR). Applies a simplified Rayleigh-only
# atmospheric correction to get approximate Rrs, then derives Chlor-a via OC4ME.
# For production work use the OLCI L2 WFR product (full C2RCC correction).

import glob
import os
import zipfile
import xarray as xr
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from matplotlib.colors import LogNorm


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


# ---- Subsampling factor (memory management) ----
# SUBSAMPLE=2 reads every 2nd row and column → quarter of the pixels.
# The native OLCI resolution is 300 m; at SUBSAMPLE=2 we get ~600 m.
# For a teaching exercise this is perfectly adequate — spatial patterns
# remain visible, and memory use drops from ~160 MB to ~40 MB per array.
# Set SUBSAMPLE=1 for full resolution if your machine has ≥ 16 GB RAM.
# Set SUBSAMPLE=4 for very low-memory machines (2–4 GB RAM).
SUBSAMPLE = 2

# ---- Unzip OLCI L1B ----
olci_path = unzip_sentinel('OL_1_EFR')

# ---- Load L1B radiances (subsampled) ----
# mask_and_scale=True applies scale_factor, add_offset and _FillValue automatically
bands = {'Oa03': 443, 'Oa04': 490, 'Oa05': 510, 'Oa06': 560}
radiances = {}
for band, wl in bands.items():
    ds = xr.open_dataset(olci_path + f'{band}_radiance.nc', mask_and_scale=True)
    arr = ds[f'{band}_radiance'].values.astype(np.float32)  # float32 halves memory vs float64
    radiances[wl] = arr[::SUBSAMPLE, ::SUBSAMPLE]
    ds.close()
    print(f"  Loaded {band} ({wl} nm): subsampled shape {radiances[wl].shape}")

# ---- Solar flux (E0 per band) ----
# E0 is the top-of-atmosphere solar irradiance for each OLCI band (mW/m²/sr/nm).
# We need it to convert radiance (L) to TOA reflectance (rho_TOA):
#   rho_TOA = pi * L / (E0 * cos(SZA))
#
# WHY netCDF4 INSTEAD OF XARRAY HERE?
#   instrument_data.nc has a dimension named 'bands' appearing twice in
#   the solar_flux variable, which triggers an xarray bug. netCDF4.Dataset
#   reads the raw array without this constraint. We then average over the
#   detector axis (axis 0 in most product versions) to get one E0 per band.
#   The shape check (E0_all.shape[0] != 21) handles products where the
#   detector and band axes are swapped.
with nc.Dataset(olci_path + 'instrument_data.nc') as ds_nc:
    solar_flux_raw = ds_nc.variables['solar_flux'][:]  # shape (detectors, bands) or (bands, detectors)
# Take mean over detector axis to get one E0 per band (21 OLCI bands)
# Axis 0 is detectors in most versions; result shape should be (21,)
if solar_flux_raw.ndim == 2:
    E0_all = np.nanmean(solar_flux_raw, axis=0)
    if E0_all.shape[0] != 21:          # try the other axis
        E0_all = np.nanmean(solar_flux_raw, axis=1)
else:
    E0_all = solar_flux_raw

band_idx = {'Oa03': 2, 'Oa04': 3, 'Oa05': 4, 'Oa06': 5}
E0 = {wl: float(E0_all[band_idx[b]]) for b, wl in bands.items()}
print("Mean solar flux E0 (mW/m2/sr/nm):", {k: f"{v:.1f}" for k, v in E0.items()})

# ---- Solar zenith angle (tie-point grid, subsampled) ----
ds_tie = xr.open_dataset(olci_path + 'tie_geometries.nc', mask_and_scale=True)
sza_tie = ds_tie['SZA'].values.astype(np.float32)
ds_tie.close()

# Upsample SZA from tie-point grid to (subsampled) full pixel grid
full_rows, full_cols = radiances[443].shape
tie_rows,  tie_cols  = sza_tie.shape
row_idx = np.round(np.linspace(0, tie_rows - 1, full_rows)).astype(int)
col_idx = np.round(np.linspace(0, tie_cols - 1, full_cols)).astype(int)
sza_full = sza_tie[np.ix_(row_idx, col_idx)]
cos_sza  = np.cos(np.radians(sza_full))
del sza_tie  # free memory

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
del qf  # free memory

# ---- Simplified Rayleigh atmospheric correction ----
#
# STEP A — Rayleigh optical depth (tau_r):
#   The formula tau_r = 0.008569 * lambda^-4 * (1 + 0.0113 * lambda^-2)
#   is an analytical approximation from Hansen & Travis (1974).
#   It depends only on wavelength (lambda in µm) and atmospheric pressure
#   (assumed standard sea-level here). tau_r is largest at short wavelengths
#   (blue light scatters more than red — Rayleigh scattering ∝ lambda^-4),
#   which is why the sky is blue.
#
# STEP B — Rayleigh reflectance (rho_r):
#   rho_r ≈ 0.75 * tau_r is a nadir-geometry approximation.
#   A full treatment would account for the solar/view geometry using
#   Rayleigh scattering phase functions — implemented in C2RCC and POLYMER.
#
# STEP C — TOA reflectance and water-leaving reflectance:
#   rho_TOA = pi * L / (E0 * cos(SZA))    [dimensionless]
#   Rrs     = (rho_TOA - rho_r) / pi      [sr-1]
#   Rrs is the "remote sensing reflectance" — the standard unit for
#   ocean colour products. It represents the ratio of water-leaving radiance
#   to the downwelling irradiance just above the surface.
#
# IMPORTANT: aerosol correction is intentionally omitted for this exercise.
# For production-quality Chlor-a, use:
#   - ESA C2RCC processor (SNAP plugin, also available as Python package)
#   - POLYMER spectral optimisation (HYGEOS, good for sun-glint scenes)
#   - Or simply download the OLCI L2 WFR product (fully corrected Rrs)
Rrs = {}
for wl in bands.values():
    lam   = wl / 1000.0                                    # nm -> um
    tau_r = 0.008569 * lam**-4 * (1 + 0.0113 * lam**-2)
    rho_r = 0.75 * tau_r
    rho_toa = (np.pi * radiances[wl]) / (E0[wl] * cos_sza)
    rrs = (rho_toa - rho_r) / np.pi
    rrs[invalid_mask] = np.nan
    rrs[rrs <= 0]     = np.nan
    Rrs[wl] = rrs

del radiances, cos_sza, invalid_mask  # free memory before OC4ME

# ---- OC4ME algorithm ----
# OC4ME = Ocean Colour 4-band Maximum band ratio Empirical algorithm.
#
# Band ratio: R = log10( max(Rrs_443, Rrs_490, Rrs_510) / Rrs_560 )
#   Taking the MAXIMUM of the blue bands makes the algorithm more robust —
#   if one band is contaminated (e.g. by sun glint), another blue band
#   still gives a reasonable estimate.
#   np.fmax() is used instead of np.maximum() because it ignores NaN
#   rather than propagating it (fmax(NaN, 0.1) = 0.1, not NaN).
#
# Polynomial: log10(Chl-a) = a0 + a1*R + a2*R^2 + a3*R^3 + a4*R^4
#   The relationship between the band ratio and Chlor-a is highly
#   nonlinear (hence the polynomial), but log10 linearises it enough.
#   The coefficients were calibrated against >2000 in-situ measurements
#   from open-ocean stations worldwide (IOCCG dataset).
#
# Final clipping: chla = np.where((chla > 0.001) & (chla < 100), chla, NaN)
#   Any algorithm output outside the physically plausible range is discarded.
#   0.001 mg/m³ ~ ultra-oligotrophic Pacific gyre
#   100 mg/m³   ~ extreme bloom (e.g. Baltic cyanobacteria in summer)
a0, a1, a2, a3, a4 = 0.3255, -2.7677, 2.4409, -1.1288, -0.4990
blue_max   = np.fmax(np.fmax(Rrs[443], Rrs[490]), Rrs[510])
R          = np.log10(blue_max / Rrs[560])
log10_chla = a0 + a1*R + a2*R**2 + a3*R**3 + a4*R**4
chla       = 10**log10_chla
chla       = np.where((chla > 0.001) & (chla < 100), chla, np.nan)

del Rrs, blue_max, R, log10_chla  # free memory

# ---- Sanity check ----
valid_frac = np.sum(np.isfinite(chla)) / chla.size * 100
print(f"Chl-a range: {np.nanmin(chla):.3f} to {np.nanmax(chla):.2f} mg/m3")
print(f"Median: {np.nanmedian(chla):.3f} mg/m3  |  Valid pixels: {valid_frac:.1f}%")

# ---- Save L2 NetCDF ----
chla_da = xr.DataArray(
    chla, dims=['y', 'x'],
    coords={'lat': (['y', 'x'], lat), 'lon': (['y', 'x'], lon)},
    attrs={
        'long_name': 'Chlorophyll-a concentration',
        'units': 'mg m-3',
        'valid_min': 0.001,
        'valid_max': 100.0,
        'processing_level': 'L2',
        'algorithm': 'OC4ME',
        'subsample_factor': SUBSAMPLE,
        'source': 'Sentinel-3 OLCI L1B, simplified Rayleigh correction',
    }
)
chla_da.to_netcdf('sentinel3_ChlorA_L2.nc')
print('L2 Chlor-a saved to sentinel3_ChlorA_L2.nc')

# ---- Map with log colour scale ----
fig = plt.figure(figsize=(12, 8))
ax  = plt.axes(projection=ccrs.PlateCarree())
valid = np.isfinite(chla)
ax.set_extent([
    np.nanmin(lon[valid]), np.nanmax(lon[valid]),
    np.nanmin(lat[valid]), np.nanmax(lat[valid])
], crs=ccrs.PlateCarree())

img = ax.pcolormesh(
    lon, lat, chla,
    norm=LogNorm(vmin=0.01, vmax=10),
    cmap=cmocean.cm.algae,
    transform=ccrs.PlateCarree(),
    rasterized=True,
)
cbar = plt.colorbar(img, ax=ax, label='Chlorophyll-a (mg/m\u00b3)', pad=0.02, shrink=0.8)
cbar.set_ticks([0.01, 0.1, 1.0, 10.0])
cbar.set_ticklabels(['0.01', '0.1', '1.0', '10.0'])
ax.add_feature(cfeature.LAND, color='lightgray', zorder=3)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=4)
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
ax.set_title(f'Chlorophyll-a — Sentinel-3 OLCI L2 (OC4ME + Rayleigh, subsample={SUBSAMPLE})',
             fontsize=12)
plt.tight_layout()
plt.savefig('ChlorA_L2_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("Map saved to ChlorA_L2_map.png")
