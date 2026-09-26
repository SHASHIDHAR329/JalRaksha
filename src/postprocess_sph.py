from pathlib import Path
import csv


PROJECT_ROOT = Path(r"C:\JalRaksha")

SPH_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "CaseDambreakVal2D_out"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sph" / "processed"


def read_semicolon_csv(path: Path):
    """Read one DualSPHysics CSV file."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        return list(reader)


def process_velocity():
    """Convert DualSPHysics velocity output into a clean CSV."""
    source = SPH_OUTPUT / "MeasuredB_Vel.csv"
    destination = OUTPUT_DIR / "velocity_timeseries.csv"

    rows = read_semicolon_csv(source)

    header = rows[1]
    data_rows = rows[2:]

    # Columns:
    # Part, Time, Vel_0.x, Vel_0.y, Vel_0.z,
    # Vel_1.x, Vel_1.y, Vel_1.z
    clean_header = [
        "part",
        "time_s",
        "vel0_x_mps",
        "vel0_y_mps",
        "vel0_z_mps",
        "vel1_x_mps",
        "vel1_y_mps",
        "vel1_z_mps",
    ]

    clean_rows = []

    for row in data_rows:
        if len(row) < 8:
            continue

        clean_rows.append([
            int(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[6]),
            float(row[7]),
        ])

    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(clean_header)
        writer.writerows(clean_rows)

    print(f"Velocity data written to: {destination}")
    print(f"Velocity rows: {len(clean_rows)}")


def process_water_surface(filename: str, output_name: str):
    """Convert a DualSPHysics water-surface gauge CSV."""
    source = SPH_OUTPUT / filename
    destination = OUTPUT_DIR / output_name

    rows = read_semicolon_csv(source)

    header = rows[0]
    data_rows = rows[1:]

    # Keep the main time and water-surface coordinates.
    clean_header = [
        "time_s",
        "swl_x_m",
        "swl_y_m",
        "swl_z_m",
    ]

    clean_rows = []

    for row in data_rows:
        if len(row) < 4:
            continue

        clean_rows.append([
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
        ])

    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(clean_header)
        writer.writerows(clean_rows)

    print(f"Water-surface data written to: {destination}")
    print(f"Water-surface rows: {len(clean_rows)}")


def main():
    print("=" * 60)
    print("JAL RAKSHA - SPH POST-PROCESSING")
    print("=" * 60)

    if not SPH_OUTPUT.exists():
        raise FileNotFoundError(
            f"SPH output directory not found:\n{SPH_OUTPUT}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    process_velocity()

    process_water_surface(
        "GaugesSWL_Swl_x02.csv",
        "water_surface_x02.csv",
    )

    process_water_surface(
        "GaugesSWL_Swl_z003.csv",
        "water_surface_z003.csv",
    )

    print("=" * 60)
    print("SPH post-processing completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()