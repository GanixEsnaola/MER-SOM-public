# =============================================================================
# explore_l1b.py
# =============================================================================
# PURPOSE:
#   Introductory demo for Block 2. Opens and inspects Sentinel-3 OLCI and
#   SLSTR L1B products, prints metadata, and produces a quick-look image.
#   Run this interactively in Jupyter or use it as a live instructor demo.
#
# SENTINEL-3 PRODUCT STRUCTURE:
#   Sentinel-3 products are delivered as .zip archives containing a .SEN3
#   folder. Inside that folder are multiple NetCDF4 (.nc) files — one per
#   variable or variable group — plus an xfdumanifest.xml manifest.
#
#   OLCI L1B (OL_1_EFR) key files:
#     Oa01_radiance.nc ... Oa21_radiance.nc   -- 21 spectral band radiances
#     geo_coordinates.nc                        -- latitude / longitude per pixel
#     tie_geometries.nc                         -- solar/view angles (subsampled)
#     qualityFlags.nc                           -- pixel quality bitmask
#     instrument_data.nc                        -- solar flux, detector info
#
#   SLSTR L1B (SL_1_RBT) key files:
#     S7_BT_in.nc, S8_BT_in.nc, S9_BT_in.nc   -- TIR brightness temperatures
#     S1_radiance_an.nc ... S6_radiance_an.nc   -- VNIR/SWIR radiances
#     geodetic_in.nc                             -- 1 km in-nadir geolocation
#     flags_in.nc                                -- cloud and quality flags
#     geometry_tn.nc                             -- angles (tie-point grid)
#
# WHY unzip_sentinel() USES GLOB?
#   Product filenames embed the acquisition datetime, orbit number, and
#   processing baseline — they are never predictable in advance. glob()
#   finds any file matching the pattern S3*OL_1_EFR*.zip regardless of
#   the exact datetime string in the filename.
#
# XARRAY vs netCDF4:
#   xr.open_dataset() gives a high-level interface with named dimensions,
#   automatic unit/attribute reading, and lazy loading (data is only read
#   from disk when you access .values). For most variables this is ideal.
#   netCDF4.Dataset() is lower-level but avoids xarray bugs with duplicate
#   dimension names (see instrument_data.nc in chlora_retrieval.py).
#
# KEY LESSON — mask_and_scale:
#   SLSTR stores brightness temperatures as scaled integers to save disk space.
#   The actual physical value = stored_integer * scale_factor + add_offset.
#   Without mask_and_scale=True, you read raw integers (~10000) instead of
#   Kelvin (~295). This is the root cause of the -58°C bug we fixed in
#   sst_retrieval.py. explore_l1b.py intentionally omits mask_and_scale on
#   the SLSTR section so students can see the raw values — then sst_retrieval.py
#   shows the correct approach.
# =============================================================================
# explore_l1b.py  (or run in Jupyter notebook)
import glob
import os
import zipfile
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

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

# ---- Unzip products ----
olci_path  = unzip_sentinel('OL_1_EFR')   # OLCI L1B
slstr_path = unzip_sentinel('SL_1_RBT')   # SLSTR L1B

# ---- OLCI L1B: Open radiance and tie-point files ----
# Open a single band (e.g. Oa08 = 665 nm, red band)
ds_rad = xr.open_dataset(olci_path + 'Oa08_radiance.nc')
print(ds_rad)  # Explore the dataset

# Radiance values (W/m²/sr/µm)
radiance = ds_rad['Oa08_radiance']
print('Shape:', radiance.shape)
print('Units:', radiance.attrs.get('units', 'N/A'))

# Open geolocation
ds_geo = xr.open_dataset(olci_path + 'geo_coordinates.nc')
lat = ds_geo['latitude'].values
lon = ds_geo['longitude'].values

# Quick look at radiance
plt.figure(figsize=(10, 6))
plt.imshow(radiance.values, cmap='viridis', vmin=0, vmax=200)
plt.colorbar(label='Radiance (mW/m²/sr/nm)')
plt.title('OLCI Band Oa08 (665 nm) — Raw L1B Radiance')
plt.savefig('olci_l1b_radiance.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- SLSTR L1B: Brightness Temperatures ----
ds_bt = xr.open_dataset(slstr_path + 'S8_BT_in.nc')  # 10.85 µm channel
BT_S8 = ds_bt['S8_BT_in'].values
print('BT S8 shape:', BT_S8.shape)