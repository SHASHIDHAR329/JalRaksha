from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(r"C:\JalRaksha")

DUALSPHYSICS_BIN = Path(
    r"C:\Users\shash\OneDrive\Desktop\JAL RAKSHA"
    r"\DualSPHysics_v5.4.3\DualSPHysics_v5.4"
    r"\bin\windows"
)

SCENARIO_NAME = "scenario_10"

SCENARIO_DIR = (
    PROJECT_ROOT
    / "simulations"
    / "dualsphysics"
    / "scenarios"
    / SCENARIO_NAME
)

XML_FILE = SCENARIO_DIR / f"{SCENARIO_NAME}_Def.xml"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "scenarios"
    / SCENARIO_NAME
)

GENCASE_EXE = DUALSPHYSICS_BIN / "GenCase_win64.exe"
GPU_EXE = DUALSPHYSICS_BIN / "DualSPHysics5.4_win64.exe"


def run_command(command: list[str], cwd: Path) -> None:
    print("\nRunning:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )


def run_scenario() -> None:
    if not XML_FILE.exists():
        raise FileNotFoundError(f"Scenario XML not found:\n{XML_FILE}")

    if not GENCASE_EXE.exists():
        raise FileNotFoundError(f"GenCase not found:\n{GENCASE_EXE}")

    if not GPU_EXE.exists():
        raise FileNotFoundError(f"DualSPHysics GPU executable not found:\n{GPU_EXE}")

    if OUTPUT_DIR.exists():
        print(f"Removing old output:\n{OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    generated_case = OUTPUT_DIR / SCENARIO_NAME

    print("=" * 60)
    print(f"RUNNING SPH SCENARIO: {SCENARIO_NAME}")
    print("=" * 60)

    # 1. Generate the initial particle files.
    run_command(
        [
            str(GENCASE_EXE),
            f"{SCENARIO_NAME}_Def",
            str(generated_case),
            "-save:all",
        ],
        cwd=SCENARIO_DIR,
    )

    # 2. Run DualSPHysics on GPU.
    run_command(
        [
            str(GPU_EXE),
            "-gpu",
            str(generated_case),
            str(OUTPUT_DIR),
        ],
        cwd=SCENARIO_DIR,
    )

    print("\nScenario completed successfully.")
    print(f"Output directory:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    run_scenario()
