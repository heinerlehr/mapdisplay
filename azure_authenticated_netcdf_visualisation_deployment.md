# Azure Deployment Instructions (North Europe, Authenticated)

## Overview

Deploy a global NetCDF/Zarr visualisation system using:

-   Azure Blob Storage (dataset)
-   Azure Container Apps (FastAPI tile API)
-   Azure Static Web Apps (frontend with authentication)
-   Microsoft Entra ID (authentication and authorization)

You can build and test everything locally before creating Azure
resources.

------------------------------------------------------------------------

## Local Preparation (Before Azure Account)

1.  Convert NetCDF → Zarr locally.
2.  Run FastAPI tile service locally.
3.  Run UI locally (Leaflet/MapLibre).
4.  Dockerize API and verify container works.
5.  (Optional) Use Azurite to emulate Blob Storage.

------------------------------------------------------------------------

## Azure Deployment Steps

### 1. Create Core Resources (Region: North Europe)

Create Resource Group: - Name: `rg-netcdf-viz-ne`

Create Storage Account (GPv2): - Create Blob Container: `data` - Upload
`mydataset.zarr/` into `data/mydataset.zarr/`

Create Azure Container Registry (ACR): - Push API Docker image.

------------------------------------------------------------------------

### 2. Deploy Backend API (Azure Container Apps)

Create Container Apps Environment: - Region: North Europe - Attach Log
Analytics Workspace

Deploy Container App: - Name: `netcdf-tiles-api` - Ingress: External -
Port: 8000 - Min replicas: 1 - Environment Variable: -
`DATASET_URL=az://data/mydataset.zarr`

------------------------------------------------------------------------

### 3. Secure API with Microsoft Entra ID

Enable Built-in Authentication ("Easy Auth"): - Provider: Microsoft
Entra ID - Require authentication for all requests.

Enable Managed Identity on Container App: - Assign Role: - Storage Blob
Data Reader - Scope: Storage Account or Container

Result: - API requires login. - API can securely read dataset from Blob.

------------------------------------------------------------------------

### 4. Deploy Frontend (Azure Static Web Apps)

Create Static Web App: - Region: North Europe - Connect GitHub
repository (recommended)

Enable Authentication: - Microsoft Entra ID provider

Add `staticwebapp.config.json`:

    {
      "routes": [
        {
          "route": "/*",
          "allowedRoles": ["authenticated"]
        }
      ]
    }

Configure frontend to call API endpoints:

-   /meta
-   /tiles/{var}/{species}/{time}/{z}/{x}/{y}.png

------------------------------------------------------------------------

### 5. Optional Performance Improvements

Add caching headers on tile responses.

Optional: - Add Azure Front Door in front of API. - Enable CDN caching
of tile URLs.

------------------------------------------------------------------------

### 6. Validation Checklist

-   Login required before access.
-   API reachable only when authenticated.
-   Map renders globally without seam.
-   Tile switching is fast (\<2 seconds).
-   Mask layer renders correctly.

------------------------------------------------------------------------

## Result

A secure, authenticated global visualisation platform supporting:

-   Variable selection
-   Species selection
-   Time navigation
-   Global pan/zoom
-   Continuous fields and boolean masks
