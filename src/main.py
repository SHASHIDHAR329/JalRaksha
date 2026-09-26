from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(r"C:\JalRaksha")
SRC_DIR = PROJECT_ROOT / "src"


STEPS = [
    "run_sph.py",
    "postprocess_sph.py",
    "run_flowtool.py",
    "prepare_hydrograph.py",
    "prepare_dflow_boundary.py",
    "run_dflowfm.py",
    "extract_flood_results.py",
]


def run_step(script_name: str) -> None:
    script = SRC_DIR / script_name

    print("\n" + "=" * 70)
    print(f"JALRAKSHA STEP: {script_name}")
    print("=" * 70)

    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with exit code {result.returncode}"
        )

    print(f"\n✓ {script_name} completed successfully.")


def main() -> None:
    print("=" * 70)
    print("JAL RAKSHA - END-TO-END PIPELINE")
    print("=" * 70)

    for step in STEPS:
        run_step(step)

    print("\n" + "=" * 70)
    print("JAL RAKSHA PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "\nFinal flood result:"
        "\nC:\\JalRaksha\\outputs\\dflowfm\\jalraksha_2d\\flood_depth_cells.csv"
    )


if __name__ == "__main__":
    main()