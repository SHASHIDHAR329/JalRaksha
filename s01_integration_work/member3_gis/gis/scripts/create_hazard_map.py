import numpy as np
import rasterio

input_file = "gis/output/test_flood_depth.tif"
output_file = "gis/output/test_hazard.tif"

with rasterio.open(input_file) as src:
    depth = src.read(1)

    hazard = np.zeros(depth.shape, dtype=np.uint8)

    hazard[(depth > 0) & (depth <= 2)] = 1
    hazard[(depth > 2) & (depth <= 3)] = 2
    hazard[depth > 3] = 3

    profile = src.profile.copy()
    profile.update(
        dtype=rasterio.uint8,
        count=1
    )

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(hazard, 1)

print("Hazard map created successfully.")
print(f"Output: {output_file}")
print("0 = No flood")
print("1 = Low hazard")
print("2 = Medium hazard")
print("3 = High hazard")