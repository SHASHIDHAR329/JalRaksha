from pathlib import Path
import csv
import json

import numpy as np
import rasterio


ROOT = Path(r"C:\JalRaksha")

SAR_DIR = ROOT / r"inputs\sar"
OUT_DIR = ROOT / r"outputs\sar_validation"

SAR_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


PREDICTION_FILE = SAR_DIR / "fno_flood_mask.tif"
SAR_MASK_FILE = SAR_DIR / "sentinel1_flood_mask.tif"


def fail(message):
    raise FileNotFoundError(message)


if not PREDICTION_FILE.exists():
    fail(
        f"FNO flood mask not found:\n{PREDICTION_FILE}\n\n"
        "Expected a binary GeoTIFF with 1=flood and 0=non-flood."
    )

if not SAR_MASK_FILE.exists():
    fail(
        f"Sentinel-1 flood mask not found:\n{SAR_MASK_FILE}\n\n"
        "Expected a binary GeoTIFF with 1=flood and 0=non-flood."
    )


with rasterio.open(PREDICTION_FILE) as pred_src:
    pred = pred_src.read(1)
    pred_transform = pred_src.transform
    pred_crs = pred_src.crs
    pred_shape = pred.shape
    pred_nodata = pred_src.nodata


with rasterio.open(SAR_MASK_FILE) as sar_src:
    sar = sar_src.read(1)
    sar_transform = sar_src.transform
    sar_crs = sar_src.crs
    sar_shape = sar.shape
    sar_nodata = sar_src.nodata


# ------------------------------------------------------------
# Geometry compatibility
# ------------------------------------------------------------

if pred_shape != sar_shape:
    raise ValueError(
        f"Raster dimensions do not match: "
        f"FNO={pred_shape}, SAR={sar_shape}"
    )

if pred_crs != sar_crs:
    raise ValueError(
        f"CRS mismatch: FNO={pred_crs}, SAR={sar_crs}"
    )

if not np.allclose(
    np.array(tuple(pred_transform)),
    np.array(tuple(sar_transform)),
    rtol=0,
    atol=1e-9
):
    raise ValueError(
        "Raster transforms do not match. "
        "Both rasters must be on the same grid."
    )


# ------------------------------------------------------------
# Binary masks
# ------------------------------------------------------------

pred_mask = pred > 0.5
sar_mask = sar > 0.5


# ------------------------------------------------------------
# Remove NoData
# ------------------------------------------------------------

valid = np.ones(pred_shape, dtype=bool)

if pred_nodata is not None:
    valid &= pred != pred_nodata

if sar_nodata is not None:
    valid &= sar != sar_nodata

pred_mask = pred_mask & valid
sar_mask = sar_mask & valid


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

true_positive = int(np.sum(pred_mask & sar_mask))
false_positive = int(np.sum(pred_mask & ~sar_mask))
false_negative = int(np.sum(~pred_mask & sar_mask))
true_negative = int(np.sum(~pred_mask & ~sar_mask))


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

intersection = true_positive
union = true_positive + false_positive + false_negative

iou = (
    intersection / union
    if union > 0
    else 1.0
)

precision = (
    true_positive / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0.0
)

recall = (
    true_positive / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0.0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0.0
)


# ------------------------------------------------------------
# Area
# ------------------------------------------------------------

pixel_width = abs(pred_transform.a)
pixel_height = abs(pred_transform.e)

pixel_area = pixel_width * pixel_height

predicted_area_m2 = float(pred_mask.sum() * pixel_area)
sar_area_m2 = float(sar_mask.sum() * pixel_area)

area_difference_percent = (
    abs(predicted_area_m2 - sar_area_m2)
    / sar_area_m2
    * 100.0
    if sar_area_m2 > 0
    else 0.0
)


# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

results = {
    "prediction_file": str(PREDICTION_FILE),
    "sar_mask_file": str(SAR_MASK_FILE),
    "crs": str(pred_crs),
    "rows": int(pred_shape[0]),
    "columns": int(pred_shape[1]),
    "pixel_width_m": pixel_width,
    "pixel_height_m": pixel_height,
    "true_positive": true_positive,
    "false_positive": false_positive,
    "false_negative": false_negative,
    "true_negative": true_negative,
    "IoU": iou,
    "precision": precision,
    "recall": recall,
    "F1": f1,
    "predicted_flood_area_m2": predicted_area_m2,
    "sar_flood_area_m2": sar_area_m2,
    "flood_area_difference_percent": area_difference_percent,
}


json_path = OUT_DIR / "sar_validation.json"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)


csv_path = OUT_DIR / "sar_validation.csv"

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results.keys()))
    writer.writeheader()
    writer.writerow(results)


print("=" * 70)
print("JALRAKSHA SAR VALIDATION ENGINE")
print("=" * 70)
print("IoU:", f"{iou:.4f}")
print("Precision:", f"{precision:.4f}")
print("Recall:", f"{recall:.4f}")
print("F1:", f"{f1:.4f}")
print(
    "Predicted flood area:",
    f"{predicted_area_m2:.2f}",
    "m²"
)
print(
    "SAR flood area:",
    f"{sar_area_m2:.2f}",
    "m²"
)
print(
    "Flood-area difference:",
    f"{area_difference_percent:.2f}%"
)
print()
print("JSON:", json_path)
print("CSV :", csv_path)
print("=" * 70)
