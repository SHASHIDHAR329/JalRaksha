import numpy as np
import rasterio

input_file = "gis/output/test_flood_depth.tif"
output_file = "gis/output/test_flood_extent.tif"

with rasterio.open(input_file) as src:

    depth = src.read(1)

    # Flooded if water depth is greater than 0
    flood_extent = (depth > 0).astype(np.uint8)

    profile = src.profile.copy()

    # Change output data type to unsigned 8-bit
    profile.update(
        dtype=rasterio.uint8,
        count=1
    )

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(flood_extent, 1)

print("Flood extent GeoTIFF created successfully.")
print(f"Output: {output_file}")
print("0 = Not flooded")
print("1 = Flooded")