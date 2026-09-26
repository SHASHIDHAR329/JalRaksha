from pathlib import Path
import re
import shutil
import subprocess
import sys
import csv

ROOT = Path(r"C:\JalRaksha")

BASE_XML = ROOT / r"simulations\dualsphysics\dam_break_validation\CaseDambreakVal2D_Def.xml"
BASE_FILEBOXES = ROOT / r"simulations\dualsphysics\dam_break_validation\CaseDambreak_FileBoxes.txt"

SPH_SCENARIO_DIR = ROOT / r"simulations\dualsphysics\scenarios"
SPH_OUTPUT_DIR = ROOT / r"outputs\sph\scenarios"

DFLOW_BASE = ROOT / r"simulations\dflowfm\jalraksha_flume"
DFLOW_DIR = ROOT / r"simulations\dflowfm"

SCRIPTS = {
    "sph": ROOT / r"src\run_sph_scenario.py",
    "flowtool": ROOT / r"src\run_flowtool_scenario.py",
    "hydro": ROOT / r"src\prepare_hydrograph_scenario.py",
    "dflow": ROOT / r"src\run_flume_scenario.py",
    "extract": ROOT / r"src\extract_flume_scenario_02.py",
}

SCENARIOS = {
    "scenario_04": (1.25, 1.50),
    "scenario_05": (1.50, 1.50),
    "scenario_06": (1.75, 1.50),
    "scenario_07": (2.00, 1.50),
    "scenario_08": (1.25, 1.75),
    "scenario_09": (1.50, 1.75),
    "scenario_10": (1.75, 1.75),
}

def run_script(path):
    print(f"\n>>> Running {path.name}")
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"{path.name} failed with code {result.returncode}")

def make_sph_scenario(name, length, height):
    out = SPH_SCENARIO_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    text = BASE_XML.read_text(encoding="utf-8")

    text = re.sub(
        r'<size x="[0-9.]+" y="2" z="[0-9.]+" />',
        f'<size x="{length}" y="2" z="{height}" />',
        text,
        count=1
    )

    (out / f"{name}_Def.xml").write_text(text, encoding="utf-8")
    shutil.copy2(BASE_FILEBOXES, out / "CaseDambreak_FileBoxes.txt")

def patch_scenario(script_path, name):
    text = script_path.read_text(encoding="utf-8")

    text = re.sub(
        r'SCENARIO_NAME\s*=\s*"scenario_\d+"',
        f'SCENARIO_NAME = "{name}"',
        text
    )

    text = re.sub(
        r'jalraksha_flume_scenario_\d+',
        f'jalraksha_flume_{name}',
        text
    )

    text = re.sub(
        r'Scenario \d+ flume extraction successful\.',
        f'{name} flume extraction successful.',
        text
    )

    script_path.write_text(text, encoding="utf-8")

def make_dflow_case(name):
    case = DFLOW_DIR / f"jalraksha_flume_{name}"

    if case.exists():
        shutil.rmtree(case)

    shutil.copytree(DFLOW_BASE, case)
    return case

def read_single_column_csv(path, column):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    values = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (ValueError, KeyError):
            pass
    return values

def collect_result(name, length, height):
    hydro = SPH_OUTPUT_DIR / name / "dam_break_hydrograph_clean.csv"
    flood = ROOT / r"outputs\dflowfm" / f"jalraksha_flume_{name}" / "flood_depth_cells.csv"

    peak_q = None
    volume = None
    max_depth = None
    mean_depth = None

    if hydro.exists():
        with hydro.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        q_values = []
        t_values = []

        for row in rows:
            try:
                t_values.append(float(row["Time [s]"]))
                q_values.append(float(row["Discharge [m3/s]"]))
            except (ValueError, KeyError):
                pass

        if q_values:
            peak_q = max(q_values)

        if len(t_values) > 1:
            volume = 0.0
            for i in range(1, len(t_values)):
                dt = t_values[i] - t_values[i - 1]
                volume += 0.5 * (q_values[i] + q_values[i - 1]) * dt

    if flood.exists():
        with flood.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        depths = []
        for row in rows:
            try:
                depths.append(float(row["max_depth_m"]))
            except (ValueError, KeyError):
                pass

        if depths:
            max_depth = max(depths)
            mean_depth = sum(depths) / len(depths)

    return {
        "scenario": name,
        "reservoir_length_m": length,
        "reservoir_height_m": height,
        "peak_Q_m3s": peak_q,
        "hydrograph_volume_m3": volume,
        "max_depth_m": max_depth,
        "mean_depth_m": mean_depth,
    }

def main():
    manifest = []

    for name, (length, height) in SCENARIOS.items():
        print("\n" + "=" * 70)
        print(f"DATASET SCENARIO: {name}")
        print(f"Reservoir length : {length:.2f} m")
        print(f"Reservoir height : {height:.2f} m")
        print("=" * 70)

        make_sph_scenario(name, length, height)

        make_dflow_case(name)

        for script in SCRIPTS.values():
            patch_scenario(script, name)

        run_script(SCRIPTS["sph"])
        run_script(SCRIPTS["flowtool"])
        run_script(SCRIPTS["hydro"])
        run_script(SCRIPTS["dflow"])
        run_script(SCRIPTS["extract"])

        manifest.append(collect_result(name, length, height))

    manifest_path = ROOT / r"outputs\dataset_manifest.csv"

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "reservoir_length_m",
                "reservoir_height_m",
                "peak_Q_m3s",
                "hydrograph_volume_m3",
                "max_depth_m",
                "mean_depth_m",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Scenarios generated : {len(manifest)}")
    print(f"Manifest             : {manifest_path}")

    for row in manifest:
        print(
            f"{row['scenario']}: "
            f"L={row['reservoir_length_m']} m, "
            f"H={row['reservoir_height_m']} m, "
            f"PeakQ={row['peak_Q_m3s']}, "
            f"MaxDepth={row['max_depth_m']}"
        )

if __name__ == "__main__":
    main()
