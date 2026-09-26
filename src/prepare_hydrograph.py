from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(r"C:\JalRaksha")

FLOWTOOL_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "processed"
    / "flowtool.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "processed"
    / "dam_break_hydrograph_clean.csv"
)


def prepare_hydrograph() -> None:
    if not FLOWTOOL_CSV.exists():
        raise FileNotFoundError(
            f"FlowTool CSV not found:\n{FLOWTOOL_CSV}"
        )

    # First 4 lines of FlowTool CSV are metadata/header information.
    df = pd.read_csv(FLOWTOOL_CSV, sep=";", skiprows=4)

    required = {"Time [s]", "InFlow_Tank???"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Required FlowTool columns missing: {sorted(missing)}"
        )

    out = pd.DataFrame(
        {
            "time_s": pd.to_numeric(df["Time [s]"]),
            "Q_raw_m3s": pd.to_numeric(df["InFlow_Tank???"]),
        }
    )

    # Preserve the initial zero.
    out["Q_clean_m3s"] = out["Q_raw_m3s"]

    # Fill isolated zero gaps after the initial sample.
    mask = (out.index > 0) & (out["Q_clean_m3s"] == 0)

    cleaned = out["Q_clean_m3s"].mask(mask)
    out["Q_clean_m3s"] = cleaned.interpolate(
        method="linear",
        limit_direction="both",
    )

    out.loc[0, "Q_clean_m3s"] = 0.0

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print("Hydrograph prepared successfully.")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Rows: {len(out)}")
    print(f"Peak Q: {out['Q_clean_m3s'].max():.6f} m3/s")
    print(
        f"Peak time: "
        f"{out.loc[out['Q_clean_m3s'].idxmax(), 'time_s']:.6f} s"
    )


if __name__ == "__main__":
    prepare_hydrograph()