from pathlib import Path
import shutil
import subprocess

PROJECT_ROOT = Path(r"C:\JalRaksha")

SCENARIO_NAME = "scenario_02"

BASE_CASE = PROJECT_ROOT / "simulations" / "dflowfm" / "jalraksha_2d"
SCENARIO_CASE = PROJECT_ROOT / "simulations" / "dflowfm" / "scenarios" / SCENARIO_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dflowfm" / "scenarios" / SCENARIO_NAME

RUN_DFLOWFM = Path(
    r"C:\Program Files\Deltares\Delft3D FM Suite 2026.01 HMWQ"
) / "plugins" / "DeltaShell.Dimr" / "kernels" / "x64" / "bin" / "run_dflowfm.bat"

MDU_FILE = SCENARIO_CASE / "jalraksha_2d.mdu"

print("Preparing scenario-specific D-Flow FM case...")
print(f"Scenario: {SCENARIO_NAME}")

# Create a separate copy of the D-Flow FM case
if SCENARIO_CASE.exists():
    shutil.rmtree(SCENARIO_CASE)

shutil.copytree(BASE_CASE, SCENARIO_CASE)

# Create separate output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read MDU
mdu_text = MDU_FILE.read_text(encoding="utf-8")

# Replace the output directory
lines = []

for line in mdu_text.splitlines():
    if line.strip().lower().startswith("outputdir"):
        lines.append(f"OutputDir = {OUTPUT_DIR}")
    else:
        lines.append(line)

MDU_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Case directory: {SCENARIO_CASE}")
print(f"Output directory: {OUTPUT_DIR}")

# Run official Deltares launcher
print("\nStarting D-Flow FM...")

result = subprocess.run(
    ["cmd.exe", "/c", "call", str(RUN_DFLOWFM), "jalraksha_2d.mdu"],
    cwd=SCENARIO_CASE,
    text=True
)

if result.returncode != 0:
    raise SystemExit(
        f"\nD-Flow FM failed with return code {result.returncode}"
    )

print("\nScenario D-Flow FM completed successfully.")
print(f"Output: {OUTPUT_DIR}")