from pathlib import Path
import subprocess


PROJECT_ROOT = Path(r"C:\JalRaksha")

FLOWTOOL_EXE = Path(
    r"C:\Users\shash\OneDrive\Desktop\JAL RAKSHA"
    r"\DualSPHysics_v5.4.3\DualSPHysics_v5.4"
    r"\bin\windows\FlowTool_win64.exe"
)

SCENARIO_NAME = "scenario_10"

SCENARIO_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "scenarios"
    / SCENARIO_NAME
)

SPH_DATA_DIR = SCENARIO_DIR / "data"

BOX_FILE = (
    PROJECT_ROOT
    / "simulations"
    / "dualsphysics"
    / "scenarios"
    / SCENARIO_NAME
    / "CaseDambreak_FileBoxes.txt"
)

OUTPUT_CSV = SCENARIO_DIR / "flowtool.csv"


def run_flowtool() -> None:
    if not FLOWTOOL_EXE.exists():
        raise FileNotFoundError(f"FlowTool not found:\n{FLOWTOOL_EXE}")

    if not SPH_DATA_DIR.exists():
        raise FileNotFoundError(f"SPH data folder not found:\n{SPH_DATA_DIR}")

    if not BOX_FILE.exists():
        raise FileNotFoundError(f"FileBoxes not found:\n{BOX_FILE}")

    command = [
        str(FLOWTOOL_EXE),
        "-dirdata",
        str(SPH_DATA_DIR),
        "-fileboxes",
        str(BOX_FILE),
        "-savecsv",
        str(OUTPUT_CSV),
        "-first:0",
        "-last:200",
        "-units:1",
    ]

    print("=" * 60)
    print(f"FLOWTOOL: {SCENARIO_NAME}")
    print("=" * 60)

    result = subprocess.run(
        command,
        cwd=SCENARIO_DIR,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FlowTool failed with exit code {result.returncode}"
        )

    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(
            f"FlowTool completed but output was not created:\n{OUTPUT_CSV}"
        )

    print("FlowTool completed successfully.")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_flowtool()
