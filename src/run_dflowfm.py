from pathlib import Path
import subprocess


DFLOW_BAT = Path(
    r"C:\Program Files\Deltares\Delft3D FM Suite 2026.01 HMWQ"
    r"\plugins\DeltaShell.Dimr\kernels\x64\bin\run_dflowfm.bat"
)

CASE_DIR = Path(r"C:\JalRaksha\simulations\dflowfm\jalraksha_2d")
MDU_FILE = CASE_DIR / "jalraksha_2d.mdu"


def run_dflowfm() -> None:
    if not DFLOW_BAT.exists():
        raise FileNotFoundError(f"D-Flow FM launcher not found: {DFLOW_BAT}")

    if not MDU_FILE.exists():
        raise FileNotFoundError(f"MDU file not found: {MDU_FILE}")

    print("Starting D-Flow FM...")
    print(f"Case directory: {CASE_DIR}")
    print(f"Model file: {MDU_FILE}")

    result = subprocess.run(
        [
            "cmd.exe",
            "/c",
            "call",
            str(DFLOW_BAT),
            MDU_FILE.name,
        ],
        cwd=CASE_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"D-Flow FM failed with exit code {result.returncode}"
        )

    print("D-Flow FM completed successfully.")


if __name__ == "__main__":
    run_dflowfm()