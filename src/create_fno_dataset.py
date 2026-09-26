from pathlib import Path
import json
import csv
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

ROOT = Path(r"C:\JalRaksha")

manifest_path = ROOT / r"outputs\dataset_manifest.csv"
out_dir = ROOT / r"outputs\fno_dataset"
out_dir.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(manifest_path)

# Common regular grid
NX = 128
NY = 16

XMIN = 0.9860695464078194
XMAX = 130.0139319438272
YMIN = 27.76290527043957
YMAX = 37.30466915202208

x_grid = np.linspace(XMIN, XMAX, NX)
y_grid = np.linspace(YMIN, YMAX, NY)
Xg, Yg = np.meshgrid(x_grid, y_grid)

# Normalization ranges from the complete 10-scenario dataset
length_min = manifest["reservoir_length_m"].min()
length_max = manifest["reservoir_length_m"].max()

height_min = manifest["reservoir_height_m"].min()
height_max = manifest["reservoir_height_m"].max()

q_min = manifest["peak_Q_m3s"].min()
q_max = manifest["peak_Q_m3s"].max()

volume_min = manifest["hydrograph_volume_m3"].min()
volume_max = manifest["hydrograph_volume_m3"].max()

inputs = []
outputs = []
scenario_ids = []

common_mask = None

def normalize(value, vmin, vmax):
    if abs(vmax - vmin) < 1e-12:
        return 0.0
    return (value - vmin) / (vmax - vmin)

for _, row in manifest.iterrows():

    scenario = row["scenario"]

    if scenario == "scenario_01":
        flood_path = (
            ROOT
            / r"outputs\dflowfm\jalraksha_flume\flood_depth_cells.csv"
        )
    else:
        flood_path = (
            ROOT
            / "outputs"
            / "dflowfm"
            / f"jalraksha_flume_{scenario}"
            / "flood_depth_cells.csv"
        )

    df = pd.read_csv(flood_path)

    points = df[["x", "y"]].to_numpy()
    depth_values = df["max_depth_m"].to_numpy()

    # Interpolate from D-Flow FM cell centres to the fixed grid
    depth_grid = griddata(
        points,
        depth_values,
        (Xg, Yg),
        method="linear",
        fill_value=np.nan
    )

    # Mask is identical for all scenarios because the mesh is identical
    mask = np.isfinite(depth_grid).astype(np.float32)

    # Outside the flume domain -> zero depth
    depth_grid = np.nan_to_num(depth_grid, nan=0.0)

    if common_mask is None:
        common_mask = mask

    # Seven input channels:
    # 0 = normalized X coordinate
    # 1 = normalized Y coordinate
    # 2 = flume mask
    # 3 = normalized reservoir length
    # 4 = normalized reservoir height
    # 5 = normalized peak discharge
    # 6 = normalized hydrograph volume

    x_norm = (Xg - XMIN) / (XMAX - XMIN)
    y_norm = (Yg - YMIN) / (YMAX - YMIN)

    length_norm = normalize(
        row["reservoir_length_m"], length_min, length_max
    )

    height_norm = normalize(
        row["reservoir_height_m"], height_min, height_max
    )

    q_norm = normalize(
        row["peak_Q_m3s"], q_min, q_max
    )

    volume_norm = normalize(
        row["hydrograph_volume_m3"], volume_min, volume_max
    )

    input_grid = np.stack(
        [
            x_norm,
            y_norm,
            mask,
            np.full_like(Xg, length_norm),
            np.full_like(Xg, height_norm),
            np.full_like(Xg, q_norm),
            np.full_like(Xg, volume_norm),
        ],
        axis=0,
    ).astype(np.float32)

    output_grid = depth_grid[None, :, :].astype(np.float32)

    inputs.append(input_grid)
    outputs.append(output_grid)
    scenario_ids.append(scenario)

X = np.stack(inputs, axis=0)
Y = np.stack(outputs, axis=0)

np.save(out_dir / "X.npy", X)
np.save(out_dir / "Y.npy", Y)
np.save(out_dir / "mask.npy", common_mask.astype(np.float32))

np.save(out_dir / "x_grid.npy", x_grid.astype(np.float32))
np.save(out_dir / "y_grid.npy", y_grid.astype(np.float32))

with (out_dir / "scenario_ids.json").open("w", encoding="utf-8") as f:
    json.dump(scenario_ids, f, indent=2)

stats = {
    "grid_nx": NX,
    "grid_ny": NY,
    "x_min": float(XMIN),
    "x_max": float(XMAX),
    "y_min": float(YMIN),
    "y_max": float(YMAX),
    "length_min": float(length_min),
    "length_max": float(length_max),
    "height_min": float(height_min),
    "height_max": float(height_max),
    "peak_Q_min": float(q_min),
    "peak_Q_max": float(q_max),
    "volume_min": float(volume_min),
    "volume_max": float(volume_max),
}

with (out_dir / "normalization.json").open("w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("=" * 70)
print("FNO DATASET CREATED")
print("=" * 70)
print("Scenarios :", len(scenario_ids))
print("Input X   :", X.shape)
print("Output Y  :", Y.shape)
print("Mask      :", common_mask.shape)
print("Grid      :", f"{NX} x {NY}")
print("Output dir:", out_dir)
print()
print("Files:")
print("  X.npy")
print("  Y.npy")
print("  mask.npy")
print("  x_grid.npy")
print("  y_grid.npy")
print("  scenario_ids.json")
print("  normalization.json")
