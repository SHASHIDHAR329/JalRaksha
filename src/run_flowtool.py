from pathlib import Path
import subprocess


PROJECT_ROOT = Path(r"C:\JalRaksha")

FLOWTOOL_EXE = Path(
    r"C:\Users\shash\OneDrive\Desktop\JAL RAKSHA"
    r"\DualSPHysics_v5.4.3\DualSPHysics_v5.4"
    r"\bin\windows\FlowTool_win64.exe"
)

SPH_DATA_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "CaseDambreakVal2D_out"
    / "data"
)

BOX_FILE = (
    PROJECT_ROOT
    / "simulations"
    / "dualsphysics"
    / "dam_break_validation"
    / "CaseDambreak_FileBoxes.txt"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "processed"
    / "flowtool.csv"
)


def run_flowtool() -> None:
    if not FLOWTOOL_EXE.exists():
        raise FileNotFoundError(
            f"FlowTool executable not found:\n{FLOWTOOL_EXE}"
        )

    if not SPH_DATA_DIR.exists():
        raise FileNotFoundError(
            f"SPH data directory not found:\n{SPH_DATA_DIR}"
        )

    if not BOX_FILE.exists():
        raise FileNotFoundError(
            f"FileBoxes file not found:\n{BOX_FILE}"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

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

    print("Starting FlowTool...")

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FlowTool failed with exit code {result.returncode}"
        )

    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(
            f"FlowTool finished but CSV was not created:\n{OUTPUT_CSV}"
        )

    print("FlowTool completed successfully.")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_flowtool()