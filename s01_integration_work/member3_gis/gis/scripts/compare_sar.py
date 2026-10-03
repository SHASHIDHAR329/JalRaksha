import numpy as np
import rasterio


MODEL_MASK = "Member3_Final/gis/output/model_flood_mask_utm.tif"
SAR_MASK = "Member3_Final/gis/output/sar_test_water_mask_utm.tif"


def read_mask(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        return data, profile


model, model_profile = read_mask(MODEL_MASK)
sar, sar_profile = read_mask(SAR_MASK)


# ---------------------------------------------------------
# Check that the rasters are on the same grid
# ---------------------------------------------------------

if model.shape != sar.shape:
    raise ValueError(
        f"Mask dimensions do not match: "
        f"model={model.shape}, SAR={sar.shape}"
    )

if model_profile["crs"] != sar_profile["crs"]:
    raise ValueError(
        f"CRS does not match: "
        f"model={model_profile['crs']}, SAR={sar_profile['crs']}"
    )

model_transform = model_profile["transform"]
sar_transform = sar_profile["transform"]

if not np.allclose(
    tuple(model_transform),
    tuple(sar_transform),
    rtol=0,
    atol=1e-10,
):
    raise ValueError(
        "Raster transforms do not match. "
        "The model and SAR masks are not on the same geographic grid."
    )


# ---------------------------------------------------------
# Convert masks to binary
# ---------------------------------------------------------

model_binary = model > 0
sar_binary = sar > 0


# ---------------------------------------------------------
# Confusion counts
# ---------------------------------------------------------

tp = int(np.sum(model_binary & sar_binary))
fp = int(np.sum(~model_binary & sar_binary))
fn = int(np.sum(model_binary & ~sar_binary))

union = tp + fp + fn

iou = tp / union if union else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall)
    else 0.0
)


# ---------------------------------------------------------
# Flooded pixel counts
# ---------------------------------------------------------

model_pixels = int(np.sum(model_binary))
sar_pixels = int(np.sum(sar_binary))


# ---------------------------------------------------------
# Flooded area
# ---------------------------------------------------------

pixel_width = abs(model_transform.a)
pixel_height = abs(model_transform.e)

if model_profile["crs"] and model_profile["crs"].is_projected:
    pixel_area_m2 = pixel_width * pixel_height
else:
    raise ValueError(
        "Area calculation requires a projected CRS in metres. "
        "Reproject the masks before running area comparison."
    )

model_area_m2 = model_pixels * pixel_area_m2
sar_area_m2 = sar_pixels * pixel_area_m2
area_difference_m2 = abs(model_area_m2 - sar_area_m2)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("SAR comparison")
print("----------------")
print("Model flooded pixels:", model_pixels)
print("SAR candidate-water pixels:", sar_pixels)

print("True positives:", tp)
print("False positives:", fp)
print("False negatives:", fn)

print("IoU:", round(float(iou), 4))
print("Precision:", round(float(precision), 4))
print("Recall:", round(float(recall), 4))
print("F1:", round(float(f1), 4))

print("Pixel area (m2):", round(float(pixel_area_m2), 4))
print("Model flooded area (m2):", round(float(model_area_m2), 2))
print("SAR flooded area (m2):", round(float(sar_area_m2), 2))
print(
    "Absolute flooded-area difference (m2):",
    round(float(area_difference_m2), 2),
)


