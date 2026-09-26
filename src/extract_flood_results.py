from pathlib import Path
import xarray as xr
import pandas as pd


def extract_flood_results(map_file: str, output_file: str) -> None:
    map_path = Path(map_file)
    output_path = Path(output_file)

    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(map_path, engine="netcdf4") as ds:
        required = {
            "waterdepth",
            "FlowElem_xcc",
            "FlowElem_ycc",
            "FlowElem_bl",
        }

        missing = required.difference(ds.variables)
        if missing:
            raise ValueError(f"Missing variables: {sorted(missing)}")

        max_depth = ds["waterdepth"].max(dim="time").values
        final_depth = ds["waterdepth"].isel(time=-1).values

        result = pd.DataFrame(
            {
                "x": ds["FlowElem_xcc"].values,
                "y": ds["FlowElem_ycc"].values,
                "bed_level_m": ds["FlowElem_bl"].values,
                "max_waterdepth_m": max_depth,
                "final_waterdepth_m": final_depth,
            }
        )

    result.to_csv(output_path, index=False)

    print(f"Created: {output_path}")
    print(f"Cells: {len(result):,}")
    print(f"Maximum depth: {result['max_waterdepth_m'].max():.3f} m")
    print(
        "Cells deeper than 0.01 m:",
        int((result["max_waterdepth_m"] > 0.01).sum()),
    )


if __name__ == "__main__":
    extract_flood_results(
        r"C:\JalRaksha\outputs\dflowfm\jalraksha_2d\jalraksha_2d_map.nc",
        r"C:\JalRaksha\outputs\dflowfm\jalraksha_2d\flood_depth_cells.csv",
    
    )