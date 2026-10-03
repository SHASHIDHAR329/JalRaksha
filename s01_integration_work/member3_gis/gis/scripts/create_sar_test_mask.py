import sys
import numpy as np
import rasterio

INPUT_FILE = "gis/data/sar/raw/sar_vv_subset.data/Sigma0_VV_dB.img"
OUTPUT_FILE = "gis/output/sar_test_water_mask.tif"

# Use a command-line threshold if provided.
# Otherwise, default to 0.20.
if len(sys.argv) > 1:
    THRESHOLD = float(sys.argv[1])
else:
    THRESHOLD = 0.20

if THRESHOLD <= 0:
    raise ValueError("Threshold must be greater than 0.")

with rasterio.open(INPUT_FILE) as src:
    sar = src.read(1)

    mask = ((sar > 0) & (sar < THRESHOLD)).astype(np.uint8)

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
    )

    with rasterio.open(OUTPUT_FILE, "w", **profile) as dst:
        dst.write(mask, 1)

candidate_pixels = int(mask.sum())
total_pixels = mask.size
percentage = candidate_pixels / total_pixels * 100

print("SAR candidate-water mask created.")
print("Input:", INPUT_FILE)
print("Output:", OUTPUT_FILE)
print("Threshold:", THRESHOLD)
print("Candidate-water pixels:", candidate_pixels)
print("Candidate-water percentage:", f"{percentage:.2f}%")