#!/bin/bash
# =============================================================================
# create_satocean_venv.sh
# =============================================================================
# PURPOSE:
#   Bootstrap the complete Python environment for the Sentinel-3 satellite
#   oceanography practical session. Run this ONCE before the session.
#
# WHAT IS A VIRTUAL ENVIRONMENT?
#   A venv is an isolated Python installation that lives inside a folder
#   (here: ./satocean/). All packages installed with pip go into that folder,
#   not into the system Python. This means:
#     - No root/sudo privileges needed
#     - No conflicts with other Python projects on the same machine
#     - Fully reproducible: you know exactly which package versions are installed
#     - Easy to delete: rm -rf satocean/ and start fresh
#
# HOW TO USE:
#   1. Run once:    bash create_satocean_venv.sh
#   2. Activate:    source satocean/bin/activate
#   3. Deactivate:  deactivate
#   4. Your prompt will show (satocean) when the venv is active
#
# WHY EMBED requirements.txt INLINE?
#   The heredoc (cat > file << 'EOF' ... EOF) writes the file right here,
#   so students only need to handle one file, not two. The requirements file
#   is deleted at the end — it is just a temporary intermediary.
#
# PLATFORM NOTE:
#   cartopy requires the system GEOS/PROJ C libraries before pip can build it.
#   On Debian/Ubuntu:  sudo apt-get install libgeos-dev libproj-dev
#   On macOS:          brew install geos proj
#   On Windows:        use conda instead, or install Visual C++ Build Tools
# =============================================================================

# Write requirements.txt inline using a "heredoc".
# Everything between << 'EOF' and EOF is treated as file content.
# The single quotes around 'EOF' prevent shell variable expansion inside.
cat > requirements.txt << 'EOF'
# Core scientific stack
numpy           # N-dimensional arrays — the foundation of all numerical work
scipy           # Scientific algorithms (statistics, signal processing, etc.)
pandas          # Tabular data, time series, CSV/Excel I/O

# NetCDF and array handling
netCDF4         # Low-level NetCDF4 reader (needed alongside xarray)
xarray          # Labelled N-D arrays — the main interface for NetCDF files
h5py            # HDF5 file support (some Sentinel products use HDF5 internally)
h5netcdf        # Alternative NetCDF4 backend via h5py

# Satellite/geospatial (cartopy needs GEOS: see note above)
cartopy         # Geographic map projections and coastline overlays
pyproj          # Coordinate reference system transformations (used by cartopy)
shapely         # Geometric operations (polygons, points) — used by cartopy
pyresample      # Resample irregular satellite swath data to regular grids

# Visualisation
matplotlib      # 2D plotting — the core plotting engine for all our maps
cmocean         # Perceptually uniform oceanographic colourmaps (thermal, algae…)
seaborn         # Statistical visualisation (used for histograms, pair plots)

# Data access
requests        # HTTP client library — used to call the Copernicus OData API
tqdm            # Progress bars for long downloads

# Optional but useful
bottleneck      # Accelerates some numpy/pandas operations (faster NaN handling)
dask            # Parallel/out-of-core computation for datasets larger than RAM

#Visulization of content netcdf files
ncdump-rich

#GEOTiff exports
rioxarray
EOF

# Create the virtual environment.
# python3 -m venv <name> creates a new folder with its own Python interpreter,
# pip, and site-packages. The environment is completely separate from the
# system Python and from any other venvs on this machine.
python3 -m venv satocean

# Activate the environment.
# After this command, 'python' and 'pip' refer to the venv's versions,
# not the system ones. Your shell prompt will change to show (satocean).
source satocean/bin/activate

# Upgrade pip first.
# The version of pip bundled with venv is often outdated. Modern pip resolves
# dependencies better and is faster at downloading packages.
pip install --upgrade pip

# Install all packages listed in requirements.txt.
# pip resolves the dependency graph, downloads wheels (pre-compiled binaries
# where available), and installs everything into satocean/lib/python3.x/
pip install -r requirements.txt

# Remove the temporary requirements file — it has done its job.
rm requirements.txt

# Print a reminder message so students know how to re-activate next time.
echo ""
echo "Setup complete. To activate the environment in future sessions:"
echo "  source satocean/bin/activate"

# =============================================================================
# CLONE THE COURSE MATERIALS REPOSITORY
# =============================================================================
# git clone downloads a full copy of a remote Git repository to your machine.
# The repository contains all the materials needed
# for the practical session. You only need an internet connection for this step;
# afterwards everything is available locally.
#
# The repository will be created as a folder called "MER-SOM-public" in the
# current directory. We then move it to the home directory (~) and rename it
# to "MER-SOM" so the path is always predictable: ~/MER-SOM.
# =============================================================================

echo ""
echo "Cloning course materials from GitHub..."
git clone https://github.com/GanixEsnaola/MER-SOM-public

# Move the cloned folder to the home directory and rename it.
# 'mv <source> <destination>' works both as a move and a rename in one step.
# $HOME is a shell variable that always expands to your home directory
# (e.g. /home/yourname on Linux, /Users/yourname on macOS).
mv MER-SOM-public "$HOME/MER-SOM"

echo ""
echo "Course materials are ready at: $HOME/MER-SOM"
