from pathlib import Path
import csv
import json

import numpy as np


ROOT = Path(r"C:\JalRaksha")

EVAL_DIR = ROOT / r"outputs\fno_model_physics\evaluation"
IMPACT_DIR = ROOT / r"outputs\impact_analysis"

IMPACT_DIR.mkdir(parents=True, exist_ok=True)

MASK = np.load(
    ROOT / r"outputs\fno_dataset\mask.npy"
).astype(bool)

x_grid = np.load(
    ROOT / r"outputs\fno_dataset\x_grid.npy"
)

y_grid = np.load(
    ROOT / r"outputs\fno_dataset\y_grid.npy"
)

dx = float(np.mean(np.diff(x_grid)))
dy = float(np.mean(np.diff(y_grid)))
cell_area = abs(dx * dy)

with open(
    ROOT / r"outputs\fno_dataset\scenario_ids.json",
    "r",
    encoding="utf-8"
) as f:
    scenario_ids = json.load(f)

with open(
    EVAL_DIR / "validation_summary.json",
    "r",
    encoding="utf-8"
) as f:
    summary = json.load(f)

scenarios = summary["validation_scenarios"]

results = []

# Prototype hazard thresholds.
# These are engineering screening classes for the demo,
# not an officially certified flood-hazard standard.

classes = [
    ("Dry", 0.00, 0.01),
    ("Low", 0.01, 0.10),
    ("Moderate", 0.10, 0.20),
    ("High", 0.20, 0.30),
    ("Severe", 0.30, float("inf")),
]

for scenario in scenarios:

    prediction_path = (
        EVAL_DIR / f"{scenario}_prediction.npy"
    )

    depth = np.load(prediction_path)[0]

    valid_depth = depth[MASK]

    row = {
        "scenario": scenario,
        "cell_area_m2": cell_area,
        "total_valid_cells": int(MASK.sum()),
    }

    for class_name, lower, upper in classes:

        selected = (
            (valid_depth >= lower)
            & (valid_depth < upper)
        )

        cells = int(selected.sum())
        area = float(cells * cell_area)

        row[f"{class_name}_cells"] = cells
        row[f"{class_name}_area_m2"] = area

    row["maximum_depth_m"] = float(valid_depth.max())
    row["mean_depth_m"] = float(valid_depth.mean())

    flooded = valid_depth >= 0.01

    row["flooded_cells_gt_0p01m"] = int(flooded.sum())
    row["flooded_area_gt_0p01m"] = float(
        flooded.sum() * cell_area
    )

    high = valid_depth >= 0.20

    row["high_severity_cells_ge_0p20m"] = int(high.sum())
    row["high_severity_area_ge_0p20m"] = float(
        high.sum() * cell_area
    )

    results.append(row)

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

csv_path = IMPACT_DIR / "flood_impact_summary.csv"

with csv_path.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)

# ------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------

json_path = IMPACT_DIR / "flood_impact_summary.json"

with json_path.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "cell_area_m2": cell_area,
            "hazard_classes": [
                {
                    "name": name,
                    "lower_m": lower,
                    "upper_m": upper,
                }
                for name, lower, upper in classes
            ],
            "scenarios": results,
        },
        f,
        indent=2
    )

# ------------------------------------------------------------
# Print report
# ------------------------------------------------------------

print("=" * 70)
print("JALRAKSHA FLOOD IMPACT ANALYSIS")
print("=" * 70)

print(
    f"Grid cell area: {cell_area:.6f} m²"
)

for row in results:

    print()
    print(row["scenario"])

    print(
        f"  Maximum depth : "
        f"{row['maximum_depth_m']:.6f} m"
    )

    print(
        f"  Mean depth    : "
        f"{row['mean_depth_m']:.6f} m"
    )

    print(
        f"  Flooded area  : "
        f"{row['flooded_area_gt_0p01m']:.2f} m²"
    )

    print(
        f"  High severity : "
        f"{row['high_severity_area_ge_0p20m']:.2f} m²"
    )

print()
print("CSV :", csv_path)
print("JSON:", json_path)
print("=" * 70)

print()
print(
    "NOTE: Hazard thresholds are prototype screening "
    "classes, not official regulatory flood-risk thresholds."
)
