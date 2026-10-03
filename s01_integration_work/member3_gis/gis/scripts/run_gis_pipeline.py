import os
import numpy as np
import rasterio


# ---------------------------------------------------------
# INPUT / OUTPUT
# ---------------------------------------------------------

INPUT_FILE = "gis/output/test_flood_depth.tif"

OUTPUT_DIR = "gis/output"

EXTENT_FILE = os.path.join(OUTPUT_DIR, "pipeline_flood_extent.tif")
HAZARD_FILE = os.path.join(OUTPUT_DIR, "pipeline_hazard.tif")
IMPACT_FILE = os.path.join(OUTPUT_DIR, "pipeline_impact.tif")


# ---------------------------------------------------------
# STEP 1: READ FLOOD DEPTH
# ---------------------------------------------------------

print("Reading flood depth map...")

with rasterio.open(INPUT_FILE) as src:

    depth = src.read(1)

    profile = src.profile.copy()

    bounds = src.bounds
    transform = src.transform
    crs = src.crs

print("Flood depth map loaded.")
print(f"Grid size: {depth.shape}")
print(f"CRS: {crs}")
print(f"Minimum depth: {depth.min()}")
print(f"Maximum depth: {depth.max()}")


# ---------------------------------------------------------
# STEP 2: CREATE FLOOD EXTENT
# ---------------------------------------------------------

print("\nCreating flood extent...")

flood_extent = (depth > 0).astype(np.uint8)

extent_profile = profile.copy()
extent_profile.update(
    dtype=rasterio.uint8,
    count=1
)

with rasterio.open(
    EXTENT_FILE,
    "w",
    **extent_profile
) as dst:

    dst.write(flood_extent, 1)

print(f"Flood extent saved to: {EXTENT_FILE}")


# ---------------------------------------------------------
# STEP 3: CREATE HAZARD MAP
# ---------------------------------------------------------

print("\nCreating hazard map...")

hazard = np.zeros(depth.shape, dtype=np.uint8)

# 0 = No flood
# 1 = Low hazard
# 2 = Medium hazard
# 3 = High hazard

hazard[(depth > 0) & (depth <= 2)] = 1
hazard[(depth > 2) & (depth <= 3)] = 2
hazard[depth > 3] = 3

hazard_profile = profile.copy()
hazard_profile.update(
    dtype=rasterio.uint8,
    count=1
)

with rasterio.open(
    HAZARD_FILE,
    "w",
    **hazard_profile
) as dst:

    dst.write(hazard, 1)

print(f"Hazard map saved to: {HAZARD_FILE}")


# ---------------------------------------------------------
# STEP 4: CREATE IMPACT MAP
# ---------------------------------------------------------

print("\nCreating impact map...")

impact = np.zeros(depth.shape, dtype=np.uint8)

# 0 = No impact
# 1 = Flooded area
# 2 = Higher-impact area

impact[(depth > 0) & (depth <= 2)] = 1
impact[depth > 2] = 2

impact_profile = profile.copy()
impact_profile.update(
    dtype=rasterio.uint8,
    count=1
)

with rasterio.open(
    IMPACT_FILE,
    "w",
    **impact_profile
) as dst:

    dst.write(impact, 1)

print(f"Impact map saved to: {IMPACT_FILE}")


# ---------------------------------------------------------
# STEP 5: BASIC STATISTICS
# ---------------------------------------------------------

flooded_pixels = int(np.sum(flood_extent == 1))
total_pixels = depth.size

print("\nGIS PIPELINE SUMMARY")
print("--------------------")
print(f"Total pixels: {total_pixels}")
print(f"Flooded pixels: {flooded_pixels}")

if total_pixels > 0:

    flooded_percentage = (
        flooded_pixels / total_pixels
    ) * 100

    print(
        f"Flooded percentage: "
        f"{flooded_percentage:.2f}%"
    )

print("\nGIS pipeline completed successfully.")