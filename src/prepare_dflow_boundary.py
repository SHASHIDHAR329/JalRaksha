from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(r"C:\JalRaksha")

HYDROGRAPH = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "processed"
    / "dam_break_hydrograph_clean.csv"
)

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "simulations"
    / "dflowfm"
    / "jalraksha_2d"
    / "boundary_conditions"
    / "JalRakshaInlet.bc"
)


def prepare_dflow_boundary() -> None:
    if not HYDROGRAPH.exists():
        raise FileNotFoundError(
            f"Hydrograph not found:\n{HYDROGRAPH}"
        )

    df = pd.read_csv(HYDROGRAPH)

    required = {"time_s", "Q_clean_m3s"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing hydrograph columns: {sorted(missing)}"
        )

    with open(BOUNDARY_FILE, "w", newline="") as f:
        f.write("[forcing]\n")
        f.write("Name = JalRakshaInlet_0001\n")
        f.write("Function = timeseries\n")
        f.write("Time-interpolation = linear\n")
        f.write("Quantity = time\n")
        f.write("Unit = seconds since 2001-01-01 00:00:00\n")
        f.write("Quantity = dischargebnd\n")
        f.write("Unit = m3/s\n")

        for time_s, flow in zip(df["time_s"], df["Q_clean_m3s"]):
            f.write(f"{time_s:.6f} {flow:.9f}\n")

    print("D-Flow FM boundary prepared successfully.")
    print(f"Boundary file: {BOUNDARY_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Peak Q: {df['Q_clean_m3s'].max():.6f} m3/s")


if __name__ == "__main__":
    prepare_dflow_boundary()