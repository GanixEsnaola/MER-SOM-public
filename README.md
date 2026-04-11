# MER-SOM — Satellite Meteorology & Oceanography

**From Raw Sentinel-3 Data to L2 Geophysical Fields**

A complete 4-hour hands-on practical course package for processing Sentinel-3 satellite data, covering Sea Surface Temperature (SST) retrieval from SLSTR and Chlorophyll-a (Chlor-a) retrieval from OLCI — from raw L1B instrument data to publication-quality maps and validation.

---

## Learning Objectives

By the end of this session, students will be able to:

- Explain satellite data processing levels (L0–L4)
- Describe the OLCI and SLSTR instruments and the physics behind ocean colour and TIR remote sensing
- Set up a reproducible Python virtual environment
- Download and open Sentinel-3 L1B data using `netCDF4`/`xarray`
- Apply calibration, quality flagging, and geophysical retrieval algorithms for SST and Chlor-a from scratch
- Produce publication-quality maps with `cartopy`/`matplotlib`
- Interpret spatial patterns in terms of marine ecosystem dynamics

**Prerequisites:** Basic Python; NetCDF familiarity helpful but not required; no prior remote sensing knowledge needed.

---

## Scientific Background

### Sentinel-3 Mission

Two satellites (Sentinel-3A: 2016, Sentinel-3B: 2018) carrying four instruments:

| Instrument | Type | Resolution | Variable |
|---|---|---|---|
| OLCI | 21-band spectrometer (400–1020 nm) | 300 m | Ocean colour, Chlor-a |
| SLSTR | Dual-view TIR/SWIR | 1 km | SST |
| SRAL | Altimeter | — | Ocean topography |
| MWR | Microwave radiometer | — | Altimetry correction |

### Data Processing Levels

`L0` (raw counts) → `L1B` (calibrated radiances/BTs) → `L2` (geophysical fields) → `L3` (gridded composites) → `L4` (analysis products)

### SST — Split-Window Algorithm (SLSTR)

```
SST = a0 + a1·BT₁₁ + a2·(BT₁₁−BT₁₂) + a3·(BT₁₁−BT₁₂)·(sec(θ)−1)
```

Coefficients: `a0=−2.6836`, `a1=1.0029`, `a2=0.8641`, `a3=0.6209`
Derived from buoy matchups (iQuam dataset). Channels: S8 (10.85 µm) and S9 (12.0 µm).

### Chlorophyll-a — OC4ME Algorithm (OLCI)

```
R = log₁₀( max(Rrs_443, Rrs_490, Rrs_510) / Rrs_560 )
log₁₀(Chlor-a) = a0 + a1·R + a2·R² + a3·R³ + a4·R⁴
```

Coefficients: `a0=0.3255`, `a1=−2.7677`, `a2=2.4409`, `a3=−1.1288`, `a4=−0.4990`

Reference: O'Reilly et al. (1998).

---

## Environment Setup

### System dependencies (Debian/Ubuntu)

```bash
sudo apt-get install libgeos-dev libproj-dev
```

### Python environment

Run the provided setup script to create a virtual environment named `satocean`:

```bash
bash create_satocean_venv.sh
```

Then activate it:

```bash
source satocean/bin/activate
```

**Key packages installed:**

| Category | Packages |
|---|---|
| Core science | numpy, scipy, pandas |
| NetCDF / arrays | xarray, netCDF4, h5py, h5netcdf |
| Geospatial | cartopy, pyproj, shapely, pyresample |
| Visualisation | matplotlib, cmocean, seaborn |
| Data access | requests, tqdm |
| Jupyter | jupyterlab, ipywidgets |
| Optional | dask, bottleneck, rioxarray, rasterio |

---

## Scripts & Workflow

Run the scripts in the order below, within the activated `satocean` environment.

### 1. `download_sentinel3.py` — Download Sentinel-3 data

Downloads products from the **Copernicus Data Space** OData API. Authenticates with username/password, searches by product type, date range, and bounding box (default: Bay of Biscay, 2023-06-01), and downloads:

- `OL_1_EFR___` — OLCI L1B (for Chlor-a and true colour)
- `SL_1_RBT___` — SLSTR L1B (for SST retrieval)
- `SL_2_WST___` — SLSTR L2 SST (for validation, optional)

```bash
# Edit credentials and region of interest inside the script before running
python download_sentinel3.py
```

### 2. `explore_l1b.py` — Inspect L1B structure *(demo/lecture)*

Opens the OLCI L1B product, loads band `Oa08` (665 nm) radiance and geolocation, makes a quick `imshow` plot. Also opens SLSTR L1B channel `S8_BT_in.nc` (10.85 µm). Designed as a live-coding demonstration.

```bash
python explore_l1b.py
# Output: olci_l1b_radiance.png
```

### 3. `sst_retrieval.py` — SLSTR L1B → SST L2

Full SST retrieval pipeline:
- Loads brightness temperatures for S8 and S9 (auto applies scale/offset)
- Applies physical BT range filter (270–320 K) and cloud masking (bit 0)
- Computes the split-window SST algorithm
- Saves SST to NetCDF and a cartopy/cmocean thermal map

```bash
python sst_retrieval.py
# Output: sentinel3_SST_L2.nc, SST_L2_map.png
```

### 4. `chlora_retrieval.py` — OLCI L1B → Chlorophyll-a L2

Chlorophyll-a retrieval pipeline:
- Loads OLCI radiances for bands at 443, 490, 510, 560 nm (subsampling configurable)
- Applies quality flagging (invalid, land, cosmetic pixels)
- Simplified Rayleigh atmospheric correction (Hansen & Travis 1974)
- Derives Chlor-a with the OC4ME polynomial
- Saves Chlor-a to NetCDF and a log-scale cartopy/cmocean algae map

```bash
python chlora_retrieval.py
# Output: sentinel3_ChlorA_L2.nc, ChlorA_L2_map.png
```

### 5. `true_color_map.py` — Quasi-true colour RGB composite

Produces an RGB image from OLCI L1B using bands at 665 nm (R), 560 nm (G), 443 nm (B). Applies TOA reflectance conversion, simplified Rayleigh correction, robust percentile clipping, and gamma correction.

```bash
python true_color_map.py
# Output: TrueColor_RGB_map.png
```

### 6. `combined_map.py` — Side-by-side SST + Chlor-a composite

Loads the two L2 NetCDF files and produces a 1×2 composite: SST with `cmocean.thermal` (left) and Chlor-a with log-scale `cmocean.algae` (right).

```bash
python combined_map.py
# Output: Sentinel3_SST_ChlorA_composite.png
```

### 7. `sst_chla_scatter.py` — SST vs Chlor-a scatter

Explores the biological relationship between thermal stratification and phytoplankton biomass. Hexbin scatter of SST vs log₁₀(Chlor-a) with reference lines at 0.1 and 1 mg m⁻³.

```bash
python sst_chla_scatter.py
# Output: SST_ChlorA_scatter.png
```

### 8. `reproject.py` — Resample to a common regular grid

Reprojects both swath fields to a common 0.01° (~1 km) regular lat/lon grid using `pyresample` nearest-neighbour. Exports co-registered CF-1.8 NetCDF files and optionally GeoTIFFs (requires `rioxarray`/`rasterio`). Produces a 2×2 validation figure comparing swath vs grid.

```bash
python reproject.py
# Output: sentinel3_SST_grid.nc, sentinel3_ChlorA_grid.nc, Sentinel3_reprojected_map.png
```

### 9. `validate_sst.py` — Validate SST against EUMETSAT L2 WST

Compares derived SST against the official SLSTR L2 WST product. Co-registers the reference using `pyresample`, computes Bias, RMSE, and Pearson R, and produces a 3-panel figure: hexbin scatter, spatial difference map, and reference SST map.

```bash
python validate_sst.py
# Output: SST_validation.png (Bias, RMSE, R printed to console)
```

---

## Output Files Summary

| File | Script | Description |
|---|---|---|
| `sentinel3_SST_L2.nc` | `sst_retrieval.py` | Derived SST (°C) on native SLSTR swath |
| `sentinel3_ChlorA_L2.nc` | `chlora_retrieval.py` | Derived Chlor-a (mg m⁻³) on native OLCI swath |
| `sentinel3_SST_grid.nc` | `reproject.py` | SST co-registered to 0.01° regular grid |
| `sentinel3_ChlorA_grid.nc` | `reproject.py` | Chlor-a co-registered to 0.01° regular grid |
| `SST_L2_map.png` | `sst_retrieval.py` | SST map (thermal colourmap) |
| `ChlorA_L2_map.png` | `chlora_retrieval.py` | Chlor-a map (log scale, algae colourmap) |
| `TrueColor_RGB_map.png` | `true_color_map.py` | OLCI quasi-true colour RGB |
| `Sentinel3_SST_ChlorA_composite.png` | `combined_map.py` | Side-by-side SST + Chlor-a |
| `SST_ChlorA_scatter.png` | `sst_chla_scatter.py` | SST vs Chlor-a hexbin scatter |
| `Sentinel3_reprojected_map.png` | `reproject.py` | Swath vs grid comparison (2×2) |
| `SST_validation.png` | `validate_sst.py` | Validation figure with Bias, RMSE, R |

---

## Data Sources

- **Copernicus Data Space:** [dataspace.copernicus.eu](https://dataspace.copernicus.eu) — Sentinel-3 L1B and L2 products (free registration required)
- **EUMETSAT:** Sentinel-3 SLSTR L2 WST reference products
- **CMEMS (Copernicus Marine Service):** Higher-level ocean analysis products
- **NASA OceanColor Web:** Ocean colour algorithms and reference data
- **iQuam (NOAA):** In-situ SST quality monitor for matchup datasets

---

## References

- O'Reilly, J.E. et al. (1998). Ocean color chlorophyll algorithms for SeaWiFS. *JGR Oceans*, 103(C11), 24937–24953.
- Merchant, C.J. et al. (2019). Satellite-based time-series of sea-surface temperature since 1981 for climate applications. *Scientific Data*, 6, 223.
- Hansen, J.E. & Travis, L.D. (1974). Light scattering in planetary atmospheres. *Space Science Reviews*, 16, 527–610.
- IOCCG (2019). *Uncertainties in Ocean Colour Remote Sensing*. Reports of the International Ocean-Colour Coordinating Group, No. 18.

---

## Session Schedule

| Block | Duration | Topic |
|---|---|---|
| 1 | ~30 min | Environment setup and data download |
| 2 | Lecture + demo | Exploring L1B data structure |
| 3 | Hands-on | SST retrieval (L1B → L2) |
| 4 | Hands-on | Chlor-a retrieval (L1B → L2) |
| 5 | Hands-on | Advanced visualisation, reprojection, validation |

---

## License

Course material and scripts are provided for educational use.
