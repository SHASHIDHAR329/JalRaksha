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

DISCHARGE_BC = (
    PROJECT_ROOT
    / "simulations"
    / "dflowfm"
    / "jalraksha_large"
    / "Discharge.bc"
)

rows = []

with HYDROGRAPH.open("r", encoding="utf-8") as f:
    next(f)

    for line in f:
        line = line.strip()

        if not line:
            continue

        time_s, discharge = line.split(",")

        rows.append(
            (float(time_s), float(discharge))
        )

with DISCHARGE_BC.open("w", encoding="utf-8") as f:
    f.write("[forcing]\n")
    f.write("Name = Boundary02_0001\n")
    f.write("Function = timeseries\n")
    f.write("Time-interpolation = linear\n")
    f.write("Quantity = time\n")
    f.write("Unit = seconds since 2001-01-01 00:00:00\n")
    f.write("Quantity = dischargebnd\n")
    f.write("Unit = m3/s\n")

    for time_s, discharge in rows:
        f.write(f"{time_s:.6f} {discharge:.9f}\n")

print("Large-domain discharge prepared successfully.")
print(f"Source scenario: {SCENARIO_NAME}")
print(f"Rows: {len(rows)}")
print(f"Output: {DISCHARGE_BC}")