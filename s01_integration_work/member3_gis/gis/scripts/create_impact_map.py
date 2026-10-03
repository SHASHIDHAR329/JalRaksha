import numpy as np
import rasterio

input_file = "gis/output/test_flood_depth.tif"
output_file = "gis/output/test_impact.tif"

with rasterio.open(input_file) as src:
    depth = src.read(1)

    impact = np.zeros(depth.shape, dtype=np.uint8)

    # Toy demonstration:
    # 0 = No impact
    # 1 = Flooded area
    # 2 = Higher-impact area
    impact[(depth > 0) & (depth <= 2)] = 1
    impact[depth > 2] = 2

    profile = src.profile.copy()
    profile.update(
        dtype=rasterio.uint8,
        count=1
    )

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(impact, 1)

print("Impact map created successfully.")
print(f"Output: {output_file}")
print("0 = No impact")
print("1 = Flooded area")
print("2 = Higher-impact area")