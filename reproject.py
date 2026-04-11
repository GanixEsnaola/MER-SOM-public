# =============================================================================
# reproject.py
# =============================================================================
# PURPOSE:
#   Reproject Sentinel-3 SST and Chlor-a from their native irregular swath
#   grids to a common regular lat/lon grid using pyresample. Saves
#   co-registered NetCDF and optionally GeoTIFF outputs.
#
# WHY REPROJECTION IS NECESSARY:
#   All Sentinel-3 L2 swath products have an IRREGULAR grid: each pixel has
#   its own lat/lon coordinate stored in a 2-D array, but array row/column
#   indices carry no geographic meaning. This means:
#
#   1. DIRECT COMPARISON is impossible:
#      SST (SLSTR, 1 km) and Chlor-a (OLCI, 300 m, or 600 m after SUBSAMPLE=2)
#      have completely different array shapes. You cannot compare them pixel-by-
#      pixel without a shared coordinate system.
#
#   2. SPATIAL OPERATIONS don't work:
#      Gradient calculation (for front detection) assumes equal-spaced pixels.
#      Area estimates require knowing the geographic size of each pixel.
#
#   3. GIS SOFTWARE expects regular grids:
#      QGIS, ArcGIS, R raster, Google Earth Engine all need georeferenced
#      rasters (GeoTIFF or CF-NetCDF with 1-D lat/lon coordinates).
#
# THE PYRESAMPLE APPROACH:
#   pyresample is the standard Python library for satellite swath resampling.
#   The workflow:
#     SwathDefinition — describes the source (irregular lat/lon per pixel)
#     AreaDefinition  — describes the target (regular grid, projection, extent)
#     kd_tree.resample_nearest() — for each target pixel, find the nearest
#         source pixel within radius_of_influence metres. No interpolation —
#         the nearest source value is assigned directly.
#
# PARAMETERS TO UNDERSTAND:
#   RESOLUTION_DEG:        target grid spacing in degrees (~1 km at mid-latitudes)
#   radius_of_influence:   search radius in metres for nearest-neighbour lookup.
#                          Rule of thumb: ~3× the source pixel size.
#                          Too small → gaps; too large → blurs fine features.
#   epsilon:               tolerance in the k-d tree search.
#                          epsilon=0.5 means "accept a neighbour up to 50%
#                          further than the true nearest" — a huge speedup with
#                          negligible effect on data quality.
#
# GEOTIFF OUTPUT (optional — requires rioxarray + rasterio):
#   GeoTIFF is the standard format for single-band georeferenced rasters.
#   EPSG:4326 = WGS84 geographic coordinate system (lat/lon in degrees).
#   rioxarray adds .rio accessor to xarray DataArrays for CRS-aware I/O.
#
# PREREQUISITES:
#   Run sst_retrieval.py and chlora_retrieval.py first.
# =============================================================================
# reproject.py
# Reprojects Sentinel-3 SST and Chlor-a L2 swath data to a common regular grid
# using pyresample. Saves co-registered GeoTIFF and NetCDF outputs.
#
# Why reprojection matters:
#   - Swath data (irregular lat/lon per pixel) cannot be directly compared pixel-by-pixel
#   - SST (SLSTR, ~1 km) and Chlor-a (OLCI, ~300 m) are on different grids
#   - Gradient/front detection, area estimates and GIS overlay all require a regular grid
#
# Outputs:
#   sentinel3_SST_grid.nc          -- SST on regular 0.01° grid
#   sentinel3_ChlorA_grid.nc       -- Chlor-a on same regular 0.01° grid
#   sentinel3_SST_grid.tif         -- GeoTIFF for GIS import
#   sentinel3_ChlorA_grid.tif      -- GeoTIFF for GIS import
#   Sentinel3_reprojected_map.png  -- Side-by-side validation figure

import glob
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from matplotlib.colors import LogNorm
from pyresample import geometry, kd_tree

# Optional: rioxarray for GeoTIFF export (pip install rioxarray rasterio)
try:
    import rioxarray  # noqa: F401
    HAS_RIO = True
except ImportError:
    HAS_RIO = False
    print("rioxarray not installed — GeoTIFF export skipped. "
          "Install with: pip install rioxarray rasterio")


# -----------------------------------------------------------------------
# 1. Auto-detect L2 NetCDF files
# -----------------------------------------------------------------------
def find_l2_netcdf(pattern, description):
    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {description} file found matching '{pattern}' in {os.getcwd()}.\n"
            f"Run sst_retrieval.py and chlora_retrieval.py first."
        )
    matches.sort(key=os.path.getmtime, reverse=True)
    print(f"Using {description}: {matches[0]}")
    return matches[0]


def load_l2(filepath):
    """Return (data, lat, lon) from an L2 NetCDF."""
    ds = xr.open_dataset(filepath)
    varname = [v for v in ds.data_vars][0]
    data = ds[varname].values.astype(np.float32)
    lat  = ds['lat'].values.astype(np.float32)
    lon  = ds['lon'].values.astype(np.float32)
    attrs = ds[varname].attrs
    ds.close()
    return data, lat, lon, attrs


sst_file = find_l2_netcdf('sentinel3_SST_L2*.nc',   'SST L2')
chl_file = find_l2_netcdf('sentinel3_ChlorA_L2*.nc', 'Chlor-a L2')

SST, lat_sst, lon_sst, sst_attrs = load_l2(sst_file)
CHL, lat_chl, lon_chl, chl_attrs = load_l2(chl_file)

print(f"SST swath shape : {SST.shape}")
print(f"Chlor-a swath shape: {CHL.shape}")


# -----------------------------------------------------------------------
# 2. Define target regular grid
# -----------------------------------------------------------------------
# We auto-compute the grid extent from the UNION of both swath extents,
# rounded outward to the nearest 0.1° to give clean grid coordinates.
# np.floor() rounds down, np.ceil() rounds up — combined they ensure the
# grid fully contains both swaths with a clean boundary.
#
# The AreaDefinition (pyresample's regular grid description) requires:
#   proj_id / projection: map projection (longlat = equirectangular, WGS84)
#   width/height:         number of columns and rows
#   area_extent:          (lon_min, lat_min, lon_max, lat_max) in degrees
#                         These are the OUTER EDGES of the corner pixels.
# The pixel centres are at RESOLUTION_DEG/2 inside the edges.
# Use the union of both swath extents so the grid covers everything
all_lons = np.concatenate([lon_sst[np.isfinite(lon_sst)].ravel(),
                           lon_chl[np.isfinite(lon_chl)].ravel()])
all_lats = np.concatenate([lat_sst[np.isfinite(lat_sst)].ravel(),
                           lat_chl[np.isfinite(lat_chl)].ravel()])

lon_min = np.floor(all_lons.min() * 10) / 10
lon_max = np.ceil (all_lons.max() * 10) / 10
lat_min = np.floor(all_lats.min() * 10) / 10
lat_max = np.ceil (all_lats.max() * 10) / 10

RESOLUTION_DEG = 0.01   # ~1 km at mid-latitudes; use 0.005 for ~500 m
n_cols = int(round((lon_max - lon_min) / RESOLUTION_DEG))
n_rows = int(round((lat_max - lat_min) / RESOLUTION_DEG))

print(f"\nTarget grid extent : {lon_min:.2f}°E – {lon_max:.2f}°E, "
      f"{lat_min:.2f}°N – {lat_max:.2f}°N")
print(f"Target grid size   : {n_cols} x {n_rows} pixels at {RESOLUTION_DEG}°")

# pyresample AreaDefinition
area_def = geometry.AreaDefinition(
    area_id    = 'sentinel3_grid',
    description= f'Regular {RESOLUTION_DEG}° lat/lon grid',
    proj_id    = 'longlat',
    projection = {'proj': 'longlat', 'datum': 'WGS84'},
    width      = n_cols,
    height     = n_rows,
    area_extent= (lon_min, lat_min, lon_max, lat_max),   # (x_ll, y_ll, x_ur, y_ur)
)

# Regular grid coordinate arrays (for saving)
grid_lons = np.linspace(lon_min + RESOLUTION_DEG/2,
                        lon_max - RESOLUTION_DEG/2, n_cols)
grid_lats = np.linspace(lat_max - RESOLUTION_DEG/2,
                        lat_min + RESOLUTION_DEG/2, n_rows)


# -----------------------------------------------------------------------
# 3. Resample both fields onto the target grid
# -----------------------------------------------------------------------
# kd_tree.resample_nearest() builds a k-d tree (k-dimensional binary search
# tree) over the source swath pixels. For each target grid pixel it finds
# the nearest source pixel in O(log N) time instead of O(N) brute force.
#
# Both SST and Chlor-a are resampled to THE SAME target AreaDefinition,
# so the resulting arrays have identical shape and identical lat/lon
# coordinates. This makes pixel-by-pixel comparison valid.
RADIUS_OF_INFLUENCE = 3000   # metres — max search radius for nearest-neighbour
                              # ~3x the native pixel size; increase for coarser grids

def resample_field(data, lat, lon, area_def, radius=RADIUS_OF_INFLUENCE):
    """Resample a swath field to the target AreaDefinition."""
    swath_def = geometry.SwathDefinition(lons=lon, lats=lat)
    gridded = kd_tree.resample_nearest(
        swath_def, data,
        area_def,
        radius_of_influence=radius,
        fill_value=np.nan,
        epsilon=0.5,          # slight approximation for speed
    )
    return gridded.astype(np.float32)


print("\nResampling SST...")
SST_grid = resample_field(SST, lat_sst, lon_sst, area_def)
print(f"  SST grid: {SST_grid.shape}, valid pixels: "
      f"{np.sum(np.isfinite(SST_grid)):,} / {SST_grid.size:,}")

print("Resampling Chlor-a...")
CHL_grid = resample_field(CHL, lat_chl, lon_chl, area_def)
print(f"  Chl grid: {CHL_grid.shape}, valid pixels: "
      f"{np.sum(np.isfinite(CHL_grid)):,} / {CHL_grid.size:,}")


# -----------------------------------------------------------------------
# 4. Save co-registered NetCDF files
# -----------------------------------------------------------------------
def save_netcdf(data, grid_lats, grid_lons, filename, varname, attrs):
    ds = xr.Dataset(
        {varname: xr.DataArray(
            data,
            dims=['lat', 'lon'],
            coords={'lat': grid_lats, 'lon': grid_lons},
            attrs={**attrs,
                   'grid_mapping': 'crs',
                   'reprojection': f'pyresample nearest-neighbour {RESOLUTION_DEG}deg'}
        )},
        attrs={
            'Conventions': 'CF-1.8',
            'source'      : 'Sentinel-3 Copernicus/ESA',
            'grid_resolution_deg': RESOLUTION_DEG,
        }
    )
    # Add CRS variable (CF convention)
    ds['crs'] = xr.DataArray(
        np.int32(0),
        attrs={'grid_mapping_name': 'latitude_longitude',
               'longitude_of_prime_meridian': 0.0,
               'semi_major_axis': 6378137.0,
               'inverse_flattening': 298.257223563}
    )
    ds.to_netcdf(filename)
    print(f"Saved: {filename}")

save_netcdf(SST_grid, grid_lats, grid_lons,
            'sentinel3_SST_grid.nc', 'sst', sst_attrs)
save_netcdf(CHL_grid, grid_lats, grid_lons,
            'sentinel3_ChlorA_grid.nc', 'chla', chl_attrs)


# -----------------------------------------------------------------------
# 5. GeoTIFF export (requires rioxarray + rasterio)
# -----------------------------------------------------------------------
if HAS_RIO:
    import rioxarray  # noqa: F811

    def save_geotiff(data, grid_lats, grid_lons, filename):
        da = xr.DataArray(
            data[np.newaxis, :, :],         # add band dimension
            dims=['band', 'y', 'x'],
            coords={
                'band': [1],
                'y': grid_lats,
                'x': grid_lons,
            }
        )
        da = da.rio.set_spatial_dims(x_dim='x', y_dim='y')
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.write_nodata(np.nan)
        da.rio.to_raster(filename)
        print(f"Saved: {filename}")

    save_geotiff(SST_grid, grid_lats, grid_lons, 'sentinel3_SST_grid.tif')
    save_geotiff(CHL_grid, grid_lats, grid_lons, 'sentinel3_ChlorA_grid.tif')
else:
    print("Skipping GeoTIFF export (rioxarray not available).")


# -----------------------------------------------------------------------
# 6. Validation figure — swath vs. grid side-by-side for one variable
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                         subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.ravel()

extent = [lon_min, lon_max, lat_min, lat_max]
feat_kw = dict(zorder=4)

# Helper
def add_map_features(ax):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color='#d4c5a9', zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, **feat_kw)
    ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5)

# SST swath (original)
add_map_features(axes[0])
v_sst = np.isfinite(SST)
img0 = axes[0].pcolormesh(lon_sst, lat_sst, SST,
    cmap=cmocean.cm.thermal,
    vmin=np.nanpercentile(SST[v_sst], 2),
    vmax=np.nanpercentile(SST[v_sst], 98),
    transform=ccrs.PlateCarree(), rasterized=True)
plt.colorbar(img0, ax=axes[0], label='SST (°C)', shrink=0.85)
axes[0].set_title('SST — Original swath (irregular grid)', fontsize=10)

# SST grid (reprojected)
add_map_features(axes[1])
lon2d, lat2d = np.meshgrid(grid_lons, grid_lats)
img1 = axes[1].pcolormesh(lon2d, lat2d, SST_grid,
    cmap=cmocean.cm.thermal,
    vmin=np.nanpercentile(SST[v_sst], 2),
    vmax=np.nanpercentile(SST[v_sst], 98),
    transform=ccrs.PlateCarree(), rasterized=True)
plt.colorbar(img1, ax=axes[1], label='SST (°C)', shrink=0.85)
axes[1].set_title(f'SST — Reprojected ({RESOLUTION_DEG}° regular grid)', fontsize=10)

# Chlor-a swath (original)
add_map_features(axes[2])
img2 = axes[2].pcolormesh(lon_chl, lat_chl, CHL,
    norm=LogNorm(vmin=0.01, vmax=10),
    cmap=cmocean.cm.algae,
    transform=ccrs.PlateCarree(), rasterized=True)
cbar2 = plt.colorbar(img2, ax=axes[2], label='Chl-a (mg/m³)', shrink=0.85)
cbar2.set_ticks([0.01, 0.1, 1, 10])
cbar2.set_ticklabels(['0.01', '0.1', '1', '10'])
axes[2].set_title('Chlor-a — Original swath (irregular grid)', fontsize=10)

# Chlor-a grid (reprojected)
add_map_features(axes[3])
img3 = axes[3].pcolormesh(lon2d, lat2d, CHL_grid,
    norm=LogNorm(vmin=0.01, vmax=10),
    cmap=cmocean.cm.algae,
    transform=ccrs.PlateCarree(), rasterized=True)
cbar3 = plt.colorbar(img3, ax=axes[3], label='Chl-a (mg/m³)', shrink=0.85)
cbar3.set_ticks([0.01, 0.1, 1, 10])
cbar3.set_ticklabels(['0.01', '0.1', '1', '10'])
axes[3].set_title(f'Chlor-a — Reprojected ({RESOLUTION_DEG}° regular grid)', fontsize=10)

fig.suptitle('Sentinel-3 L2: Swath geometry vs Regular grid\n'
             '(pyresample nearest-neighbour resampling)', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('Sentinel3_reprojected_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: Sentinel3_reprojected_map.png")
print("\nOutputs ready for GIS / further analysis:")
print("  sentinel3_SST_grid.nc      — SST on regular grid (CF-1.8 NetCDF)")
print("  sentinel3_ChlorA_grid.nc   — Chlor-a on regular grid (CF-1.8 NetCDF)")
if HAS_RIO:
    print("  sentinel3_SST_grid.tif     — SST GeoTIFF (EPSG:4326)")
    print("  sentinel3_ChlorA_grid.tif  — Chlor-a GeoTIFF (EPSG:4326)")
