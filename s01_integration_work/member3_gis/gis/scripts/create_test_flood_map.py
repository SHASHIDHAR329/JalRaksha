import numpy as np
import rasterio
from rasterio.transform import from_origin

# Create a 100 x 100 flood-depth grid
height = 100
width = 100

data = np.zeros((height, width), dtype=np.float32)

# Create a simple synthetic flooded region
data[30:70, 35:65] = 2.0

# Add a smaller deeper area
data[45:60, 45:55] = 4.0

# Geographic location of the top-left corner
west = 76.60
north = 12.40

# Pixel size in geographic coordinates
pixel_size = 0.001

transform = from_origin(
    west,
    north,
    pixel_size,
    pixel_size
)

# Coordinate Reference System
crs = "EPSG:4326"

output_file = "gis/output/test_flood_depth.tif"

with rasterio.open(
    output_file,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=data.dtype,
    crs=crs,
    transform=transform,
) as dst:
    dst.write(data, 1)

print("Test flood GeoTIFF created successfully.")
print(f"Output: {output_file}")
print(f"CRS: {crs}")
print(f"Grid size: {width} x {height}")