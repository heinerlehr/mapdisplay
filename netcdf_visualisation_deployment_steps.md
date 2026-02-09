# Deployment Steps --- Global NetCDF Visualisation Service (Azure)

## Objective

Deploy a web application that allows interactive visualisation (map +
time + species + variable selection) of a global dataset with
dimensions:

-   time (24 + average)
-   lat (720)
-   lon (1440)
-   species (2)
-   \~5--6 continuous variables + 1 boolean mask

------------------------------------------------------------------------

## High-Level Architecture

**Storage** - Dataset stored as Zarr in Azure Blob Storage.

**Backend** - FastAPI service that: - Reads Zarr via Xarray. - Serves
Web Mercator tiles (XYZ). - Provides metadata endpoint.

**Frontend** - Static Leaflet or MapLibre app hosted separately.

**Deployment** - Azure Container Apps (API) - Azure Blob Storage (+
optional CDN) (data or precomputed tiles) - Azure Static Web Apps (UI)

------------------------------------------------------------------------

## Step-by-Step Implementation

### 1. Prepare Dataset

1.  Load original NetCDF.
2.  Ensure longitude is in \[-180, 180) and sorted.
3.  Convert to Zarr with appropriate chunking:

-   time: 1
-   species: 1
-   lat/lon: \~256x256

4.  Save as:

mydataset.zarr/

------------------------------------------------------------------------

### 2. Upload Data to Azure

1.  Create Azure Storage Account.
2.  Create Blob Container.
3.  Upload entire mydataset.zarr directory.
4.  Configure access:
    -   Preferred: Managed Identity + RBAC
    -   Alternative: Storage key / SAS

------------------------------------------------------------------------

### 3. Implement Backend API (FastAPI)

Core endpoints:

#### /meta

Returns: - Variables - Species labels - Time index (+ "avg") - Lat/lon
bounds - Variable types (continuous vs boolean) - Recommended maxZoom=4

#### /tiles/{var}/{species}/{time}/{z}/{x}/{y}.png

Processing pipeline:

1.  Select (var, species, time) slice.
2.  Compute tile bounding box (Web Mercator).
3.  Interpolate to 256×256 grid:
    -   Continuous → bilinear
    -   Mask → nearest
4.  Apply colormap or mask transparency.
5.  Return PNG with caching headers.

Add caching: - Cache slices (\~300 total). - Optional tile LRU cache.

------------------------------------------------------------------------

### 4. Containerise Backend

Docker image includes:

-   fastapi
-   uvicorn
-   xarray
-   zarr
-   fsspec
-   adlfs
-   numpy
-   matplotlib

Expose API on port 8000.

------------------------------------------------------------------------

### 5. Deploy Backend to Azure Container Apps

1.  Push Docker image to Azure Container Registry.
2.  Create Container App.
3.  Configure environment variables:
    -   Dataset location (Blob URL).
    -   Credentials or Managed Identity.
4.  Enable public HTTPS endpoint.

------------------------------------------------------------------------

### 6. Implement Frontend

Static web application with:

-   Leaflet or MapLibre map.
-   Dropdown: variable.
-   Dropdown: species.
-   Slider: time (+ avg).
-   Tile layer URL pattern:

/tiles/{var}/{species}/{time}/{z}/{x}/{y}.png

Recommended settings:

-   maxZoom = 4
-   Tile opacity control.

------------------------------------------------------------------------

### 7. Deploy Frontend

Host on:

-   Azure Static Web Apps (recommended)

Configure API base URL.

------------------------------------------------------------------------

### 8. Optional Performance Optimisation

**Option A --- Dynamic tiles** - Keep server-side cache enabled. - Add
CDN in front of API.

**Option B --- Precompute tiles** - Generate tiles for all (var,
species, time) combinations. - Upload tiles to Blob Storage. - Serve via
CDN.

------------------------------------------------------------------------

### 9. Validation Checklist

-   Map renders globally without dateline seam.
-   Time/species switching \< 1--2 seconds.
-   Mask overlay behaves correctly.
-   Color scaling appropriate.
-   Tile caching headers active.

------------------------------------------------------------------------

## Result

A scalable global visualisation system where users can:

-   Pan/zoom globally.
-   Switch variables.
-   Switch species.
-   Move through time or view averages.
-   View continuous fields and boolean masks correctly.
