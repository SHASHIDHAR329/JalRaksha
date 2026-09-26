from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(r"C:\JalRaksha")

SPH_CASE = (
    PROJECT_ROOT
    / "simulations"
    / "dualsphysics"
    / "dam_break_validation"
)

SPH_BATCH = SPH_CASE / "wCaseDambreakVal2D_win64_GPU.bat"

SPH_CASE_OUTPUT = SPH_CASE / "CaseDambreakVal2D_out"

PROJECT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "CaseDambreakVal2D_out"
)


def run_sph() -> None:
    if not SPH_BATCH.exists():
        raise FileNotFoundError(
            f"DualSPHysics batch file not found:\n{SPH_BATCH}"
        )

    print("Starting DualSPHysics...")

    # Remove previous case output so the batch file does not stop
    # and ask for permission to delete old results.
    if SPH_CASE_OUTPUT.exists():
        print(f"Removing previous SPH output: {SPH_CASE_OUTPUT}")
        shutil.rmtree(SPH_CASE_OUTPUT)

    result = subprocess.run(
        [
            "cmd.exe",
            "/c",
            "call",
            SPH_BATCH.name,
        ],
        cwd=SPH_CASE,
        text=True,
        input="1\n",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"DualSPHysics failed with exit code {result.returncode}"
        )

    if not SPH_CASE_OUTPUT.exists():
        raise FileNotFoundError(
            f"DualSPHysics finished, but expected output was not found:\n"
            f"{SPH_CASE_OUTPUT}"
        )

    PROJECT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if PROJECT_OUTPUT.exists():
        shutil.rmtree(PROJECT_OUTPUT)

    shutil.copytree(SPH_CASE_OUTPUT, PROJECT_OUTPUT)

    print("DualSPHysics completed successfully.")
    print(f"SPH output copied to:\n{PROJECT_OUTPUT}")


if __name__ == "__main__":
    run_sph()