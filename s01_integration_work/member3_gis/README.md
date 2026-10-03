# JalRaksha - Member 3 GIS + SAR Package

## Scope

This package contains the GIS and SAR processing work developed for JalRaksha.

Member 3 scope:
- GIS flood layers
- Flood extent generation
- Hazard classification
- Impact-map generation
- Sentinel-1 SAR processing support
- SAR candidate-water mask generation
- SAR/model comparison utilities

The package does not contain dashboard or presentation work.

## GIS

Main pipeline:

gis/scripts/run_gis_pipeline.py

The current GIS outputs use controlled/test flood-depth data and demonstrate the processing workflow.

## SAR

A Sentinel-1 GRD scene was processed using ESA SNAP.

Processing included:
1. Precise orbit application
2. VV/VH calibration
3. Terrain correction using Copernicus 30 m DEM
4. WGS84 geographic output
5. Controlled VV subset creation

The reusable subset is:

gis/data/sar/raw/sar_vv_subset.dim

The original large SAFE product and full-scene terrain-corrected data are not included in this handoff package.

## SAR candidate-water mask

Script:

gis/scripts/create_sar_test_mask.py

The current threshold is experimental.

The resulting SAR mask must NOT be described as a validated flood observation.

## SAR/model comparison

Script:

gis/scripts/compare_sar.py

The comparison checks:
- dimensions
- CRS
- geographic grid alignment
- true positives
- false positives
- false negatives
- IoU
- Precision
- Recall
- F1
- flooded area
- flooded-area difference

The integrated model flood mask will be supplied later by the project integration.

## Supporting modules

impact/impact.py
trust/trust_engine.py
validation/metrics.py

These provide impact calculations, trust checks, and validation metrics.

## Environment

The original virtual environment is not included.

Install dependencies with:

pip install -r requirements.txt
