from pathlib import Path
import xarray as xr
import pandas as pd

PROJECT_ROOT = Path(r"C:\JalRaksha")

MAP_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dflowfm"
    / "jalraksha_large"
    / "westernscheldt06_map.nc"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "dflowfm"
    / "jalraksha_large"
    / "flood_depth_cells.csv"
)

ds = xr.open_dataset(MAP_FILE)

print("Variables found:")
for name in ds.variables:
    print(f"  {name}")

# Locate water depth variable
if "waterdepth" not in ds:
    raise RuntimeError("waterdepth variable was not found in the map file.")

depth = ds["waterdepth"]

# Find cell coordinates
if "FlowElem_xcc" in ds and "FlowElem_ycc" in ds:
    x = ds["FlowElem_xcc"]
    y = ds["FlowElem_ycc"]
elif "mesh2d_face_x" in ds and "mesh2d_face_y" in ds:
    x = ds["mesh2d_face_x"]
    y = ds["mesh2d_face_y"]
else:
    raise RuntimeError("Could not find cell-center coordinates.")

# Maximum depth reached at each cell
max_depth = depth.max(dim="time")

df = pd.DataFrame({
    "cell": range(max_depth.size),
    "x": x.values,
    "y": y.values,
    "max_depth_m": max_depth.values
})

df.to_csv(OUTPUT_CSV, index=False)

print()
print(f"Created: {OUTPUT_CSV}")
print(f"Cells: {len(df)}")
print(f"Maximum depth: {df['max_depth_m'].max():.3f} m")
print(
    f"Cells deeper than 0.01 m: "
    f"{(df['max_depth_m'] > 0.01).sum()}"
)
print(
    f"Cells deeper than 0.10 m: "
    f"{(df['max_depth_m'] > 0.10).sum()}"
)