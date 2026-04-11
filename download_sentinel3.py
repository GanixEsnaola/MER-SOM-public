# =============================================================================
# download_sentinel3.py
# =============================================================================
# PURPOSE:
#   Authenticate with the Copernicus Data Space Ecosystem and download
#   Sentinel-3 products programmatically using the OData REST API.
#
# COPERNICUS DATA SPACE:
#   The EU's free satellite data portal (dataspace.copernicus.eu).
#   All Sentinel-1/2/3/5P data are freely available after registration.
#   The OData API lets us search and download data without a browser.
#
# AUTHENTICATION (OAuth2 password flow):
#   The portal uses OAuth2 tokens. We POST our username+password to the
#   identity server and receive a short-lived Bearer token. This token is
#   then included in the Authorization header of every API request.
#   getpass.getpass() prompts for the password without echoing it to the
#   terminal — never hard-code credentials in a script!
#
# ODATA SEARCH QUERY:
#   The catalogue URL accepts OData filter expressions. Our filter:
#     Collection/Name eq 'SENTINEL-3'          -- mission
#     + productType attribute filter            -- instrument+level (e.g. OL_1_EFR___)
#     + ContentDate/Start ge <date>            -- temporal constraint
#     + OData.CSC.Intersects(area=<polygon>)   -- spatial constraint (WKT polygon)
#   The trailing underscores in product types (OL_1_EFR___) are part of
#   the official EUMETSAT product naming convention — do not omit them.
#
# PRODUCTS TO DOWNLOAD:
#   OL_1_EFR___  OLCI L1B Full Resolution radiances → used for Chlor-a + RGB
#   SL_1_RBT___  SLSTR L1B Brightness Temperatures  → used for SST
#   SL_2_WST___  SLSTR L2 SST reference product     → used for validation
#   (OL_2_WFR___ OLCI L2 with C2RCC-corrected Rrs   → alternative for Chlor-a)
#
# HOW TO CUSTOMISE:
#   Change START_DATE / END_DATE (ISO 8601 format)
#   Change BBOX (WKT POLYGON with lon lat pairs, last point = first point)
#   Add/remove entries in PRODUCTS_TO_DOWNLOAD
#
# OUTPUTS:
#   One .zip file per product, named exactly as the product on the server.
#   E.g.: S3B_OL_1_EFR____20230601T104426_...004.SEN3.zip
#   The downstream scripts auto-detect these files by glob pattern.
# =============================================================================

# getpass provides a secure password prompt that does not echo characters
import getpass
import os
import requests   # HTTP library for REST API calls
from tqdm import tqdm  # Progress bar for downloads


def get_access_token():
    """
    Authenticate with Copernicus Data Space and return an OAuth2 Bearer token.

    PEDAGOGICAL NOTE:
    OAuth2 password flow:
      1. Client sends username + password to the token endpoint (HTTPS POST).
      2. Server validates credentials and returns a JSON response containing
         'access_token' (short-lived, ~10 min) and 'refresh_token'.
      3. Client includes the access token in the Authorization header of
         subsequent requests: 'Authorization: Bearer <token>'.

    This is safer than Basic Auth (credentials sent every request) because
    the token can be revoked server-side without changing the password.
    """
    username = input("Copernicus Data Space email: ")
    password = getpass.getpass("Password: ")  # Reads without echoing to terminal

    # Token endpoint for Copernicus Data Space identity server
    url = ('https://identity.dataspace.copernicus.eu/auth/realms/'
           'CDSE/protocol/openid-connect/token')

    # POST body for OAuth2 Resource Owner Password Credentials Grant
    data = {
        'client_id': 'cdse-public',   # Public client — no client secret needed
        'grant_type': 'password',      # We're using username+password directly
        'username': username,
        'password': password,
    }
    r = requests.post(url, data=data)
    r.raise_for_status()  # Raises HTTPError for 4xx/5xx responses
    return r.json()['access_token']


def search_products(token, product_type, start_date, end_date, bbox):
    """
    Search the Copernicus OData catalogue for Sentinel-3 products.

    PEDAGOGICAL NOTE — OData filter syntax:
    OData (Open Data Protocol) is a REST-based query standard, similar to
    SQL but for web APIs. Our filter is built from these clauses:

      Collection/Name eq 'SENTINEL-3'
          Selects the Sentinel-3 mission collection.

      Attributes/OData.CSC.StringAttribute/any(att:...)
          Filters on a product attribute by name+value. This is how we
          select OL_1_EFR___ vs SL_1_RBT___ etc.
          The trailing underscores are part of the official product type name.

      ContentDate/Start ge <datetime>Z
          ContentDate/End le <datetime>Z
          Temporal filter. The 'Z' suffix means UTC (Zulu time).

      OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((...))'))
          Spatial filter. WKT polygon in EPSG:4326 (decimal degrees).
          Points are (longitude latitude), NOT (latitude longitude).
          The polygon must be closed (last point = first point).

    Returns up to $top=5 results, newest first.
    """
    base = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'
    query = (
        f"?$filter=Collection/Name eq 'SENTINEL-3'"
        f" and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}')"
        f" and ContentDate/Start ge {start_date}Z"
        f" and ContentDate/End le {end_date}Z"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{bbox}')"
        "&$top=5"
    )
    headers = {'Authorization': f'Bearer {token}'}
    r = requests.get(base + query, headers=headers)
    r.raise_for_status()
    results = r.json().get('value', [])
    print(f"  Found {len(results)} result(s) for {product_type}")
    for p in results:
        print(f"    - {p['Name']}")
    return results


def download_product(token, product_id, filename):
    """
    Download a single Sentinel-3 product by its OData ID.

    PEDAGOGICAL NOTE — Streamed download:
    Large satellite products can be several GB. Downloading them all into
    memory before saving would crash on small VMs. Instead we use
    stream=True, which keeps the connection open and delivers the response
    body in small chunks (chunk_size=8192 bytes = 8 KB).

    We also read Content-Length from the response headers to initialise the
    tqdm progress bar with the correct total size.

    The 'Already exists' check means this function is idempotent — safe to
    run multiple times; it only downloads what is missing.
    """
    if os.path.exists(filename):
        print(f"  Already exists, skipping: {filename}")
        return

    # Download endpoint: /odata/v1/Products(<uuid>)/$value
    url = (f'https://download.dataspace.copernicus.eu/odata/v1/'
           f'Products({product_id})/$value')
    headers = {'Authorization': f'Bearer {token}'}

    # stream=True: don't load the entire response into memory
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))

    with open(filename, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  Downloaded: {filename}")


if __name__ == '__main__':
    # ── Configuration ── edit these for your scene of interest ──────────────
    START_DATE = '2023-06-01T00:00:00'
    END_DATE   = '2023-06-01T23:59:59'

    # WKT bounding polygon: Bay of Biscay (lon lat pairs, closed ring)
    # Approximate extent: 10°W–5°E, 35°N–48°N
    BBOX = 'POLYGON((-10 35, 5 35, 5 48, -10 48, -10 35))'

    # Dictionary of products to download.
    # Keys are the exact productType strings used by the OData API.
    # The trailing underscores are mandatory — they pad the string to a fixed
    # 12-character field per the EUMETSAT product naming convention.
    PRODUCTS_TO_DOWNLOAD = {
        'OL_1_EFR___': 'OLCI L1B Full Resolution (needed for Chlor-a, true colour)',
        'SL_1_RBT___': 'SLSTR L1B Brightness Temperatures (needed for SST)',
        'SL_2_WST___': 'SLSTR L2 Sea Surface Temperature (needed for validation)',
        # Uncomment the line below to get fully atmosphere-corrected Rrs directly
        # instead of computing it from L1B in chlora_retrieval.py:
        # 'OL_2_WFR___': 'OLCI L2 Water Full Resolution (C2RCC Rrs, Chlor-a reference)',
    }

    # Authenticate once; token is reused for all searches and downloads
    TOKEN = get_access_token()

    for product_type, description in PRODUCTS_TO_DOWNLOAD.items():
        print(f"\nSearching for {description}...")
        results = search_products(TOKEN, product_type, START_DATE, END_DATE, BBOX)
        if not results:
            print(f"  No products found — try adjusting the date or bounding box.")
            continue
        p = results[0]  # Download the first (best-matching) result
        download_product(TOKEN, p['Id'], p['Name'] + '.zip')
