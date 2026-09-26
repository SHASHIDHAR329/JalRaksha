from pathlib import Path
import xarray as xr
import pandas as pd

PROJECT_ROOT = Path(r"C:\JalRaksha")
SCENARIO_NAME = "scenario_02"

MAP_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dflowfm"
    / "jalraksha_large_scenario_02"
    / "westernscheldt06_map.nc"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "dflowfm"
    / "jalraksha_large_scenario_02"
    / "flood_depth_cells.csv"
)

ds = xr.open_dataset(MAP_FILE)

depth = ds["waterdepth"]

max_depth = depth.max(dim="time")

x = ds["FlowElem_xcc"]
y = ds["FlowElem_ycc"]

df = pd.DataFrame({
    "cell": range(max_depth.size),
    "x": x.values,
    "y": y.values,
    "max_depth_m": max_depth.values
})

df.to_csv(OUTPUT_CSV, index=False)

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