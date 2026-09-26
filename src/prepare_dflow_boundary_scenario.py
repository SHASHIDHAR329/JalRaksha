from pathlib import Path

PROJECT_ROOT = Path(r"C:\JalRaksha")
SCENARIO_NAME = "scenario_02"

HYDROGRAPH = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "scenarios"
    / SCENARIO_NAME
    / "dam_break_hydrograph_clean.csv"
)

BC_FILE = (
    PROJECT_ROOT
    / "simulations"
    / "dflowfm"
    / "jalraksha_2d"
    / "boundary_conditions"
    / "JalRakshaInlet.bc"
)

rows = []

with HYDROGRAPH.open("r", encoding="utf-8") as f:
    next(f)  # skip header

    for line in f:
        line = line.strip()

        if not line:
            continue

        time_s, discharge = line.split(",")

        rows.append(
            (float(time_s), float(discharge))
        )

with BC_FILE.open("w", encoding="utf-8") as f:
    f.write("[forcing]\n")
    f.write("Name = JalRakshaInlet_0001\n")
    f.write("Function = timeseries\n")
    f.write("Time-interpolation = linear\n")
    f.write("Quantity = time\n")
    f.write("Unit = seconds since 2001-01-01 00:00:00\n")
    f.write("Quantity = dischargebnd\n")
    f.write("Unit = m3/s\n")

    for time_s, discharge in rows:
        f.write(f"{time_s:.6f} {discharge:.9f}\n")

print("D-Flow FM boundary condition prepared successfully.")
print(f"Scenario: {SCENARIO_NAME}")
print(f"Rows written: {len(rows)}")
print(f"Boundary file: {BC_FILE}")