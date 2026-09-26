from pathlib import Path

import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(r"C:\JalRaksha")

MAP_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dflowfm"
    / "jalraksha_flume"
    / "tidalflume_map.nc"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "dflowfm"
    / "jalraksha_flume"
    / "flood_depth_cells.csv"
)


print("Opening D-Flow FM result...")
ds = xr.open_dataset(MAP_FILE)

depth = ds["mesh2d_waterdepth"]

x = ds["mesh2d_face_x"].values
y = ds["mesh2d_face_y"].values

# Maximum water depth reached at each cell
max_depth = depth.max(dim="time").values

df = pd.DataFrame({
    "cell": range(len(max_depth)),
    "x": x,
    "y": y,
    "max_depth_m": max_depth,
})

df.to_csv(OUTPUT_CSV, index=False)

print()
print("Flume flood extraction successful.")
print(f"Cells: {len(df)}")
print(f"Maximum depth: {df['max_depth_m'].max():.6f} m")
print(f"Mean depth: {df['max_depth_m'].mean():.6f} m")
print(
    f"Cells > 0.01 m: "
    f"{(df['max_depth_m'] > 0.01).sum()}"
)
print(
    f"Cells > 0.10 m: "
    f"{(df['max_depth_m'] > 0.10).sum()}"
)
print(f"Output: {OUTPUT_CSV}")