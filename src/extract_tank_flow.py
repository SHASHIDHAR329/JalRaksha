from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\JalRaksha")

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "processed"
    / "flowtool.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "processed"
    / "tank_flow_diagnostic.csv"
)


def main():
    print("=" * 60)
    print("JAL RAKSHA - FLOWTOOL TANK FLOW EXTRACTION")
    print("=" * 60)

    df = pd.read_csv(
        INPUT_FILE,
        sep=";",
        skiprows=4,
    )

    result = df[
        [
            "Time [s]",
            "InFlow_Tank???",
            "OutFlow_Tank???",
        ]
    ].copy()

    result.columns = [
        "time_s",
        "inflow_tank",
        "outflow_tank",
    ]

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Rows: {len(result)}")
    print(f"Start time: {result['time_s'].min():.6f} s")
    print(f"End time: {result['time_s'].max():.6f} s")
    print(
        f"Maximum Tank inflow: "
        f"{result['inflow_tank'].max():.6f}"
    )
    print(
        f"Maximum Tank outflow: "
        f"{result['outflow_tank'].max():.6f}"
    )

    print(f"\nSaved to:\n{OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()