from pathlib import Path
import csv
import os
import re
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(r"C:\JalRaksha")

SCENARIO_NAME = "scenario_10"

CASE_DIR = (
    PROJECT_ROOT
    / "simulations"
    / "dflowfm"
    / f"jalraksha_flume_{SCENARIO_NAME}"
)

MDU_FILE = CASE_DIR / "tidalflume.mdu"
BC_FILE = CASE_DIR / "Discharge.bc"
CACHE_FILE = CASE_DIR / "tidalflume.cache"

HYDROGRAPH_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "scenarios"
    / SCENARIO_NAME
    / "dam_break_hydrograph_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dflowfm"
    / f"jalraksha_flume_{SCENARIO_NAME}"
)

RUN_BAT = Path(
    r"C:\Program Files\Deltares\Delft3D FM Suite 2026.01 HMWQ"
) / "plugins" / "DeltaShell.Dimr" / "kernels" / "x64" / "bin" / "run_dflowfm.bat"

CMD_FILE = CASE_DIR / "run_flume_scenario.cmd"
LOG_FILE = CASE_DIR / "run.log"


def require_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} not found:\n{path}")


def remove_keys(lines, keys):
    keys = {k.lower() for k in keys}
    result = []

    for line in lines:
        if "=" in line:
            key = line.split("=", 1)[0].strip().lower()
            if key in keys:
                continue

        result.append(line)

    return result


def set_value(lines, key, value):
    pattern = re.compile(
        rf"^\s*{re.escape(key)}\s*=",
        re.IGNORECASE,
    )

    result = []
    found = False

    for line in lines:
        if pattern.match(line):
            result.append(f"{key} = {value}")
            found = True
        else:
            result.append(line)

    if not found:
        result.append(f"{key} = {value}")

    return result


def find_proj_db():
    root = Path(
        r"C:\Program Files\Deltares\Delft3D FM Suite 2026.01 HMWQ"
    )

    matches = list(root.rglob("proj.db"))

    return matches[0] if matches else None


print("=" * 70)
print(f"JALRAKSHA FLUME - {SCENARIO_NAME.upper()}")
print("=" * 70)

require_file(CASE_DIR, "Scenario case directory")
require_file(MDU_FILE, "MDU file")
require_file(HYDROGRAPH_FILE, "Scenario hydrograph")
require_file(RUN_BAT, "D-Flow FM launcher")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Backup
# ------------------------------------------------------------

print("\n[1] Creating backup...")

shutil.copy2(
    MDU_FILE,
    CASE_DIR / "tidalflume.mdu.scenario_backup.bak"
)

if BC_FILE.exists():
    shutil.copy2(
        BC_FILE,
        CASE_DIR / "Discharge.bc.scenario_backup.bak"
    )

# ------------------------------------------------------------
# Clean MDU
# ------------------------------------------------------------

print("[2] Cleaning MDU...")

mdu_lines = MDU_FILE.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines()

mdu_lines = remove_keys(
    mdu_lines,
    {
        "transportmethod",
        "writebalancefile",
        "layertype",
        "numtopsig",
        "sigmagrowthfactor",
        "qhrelax",
        "jaorgsethu",
        "effectspiral",
        "wavenikuradse",
        "trtdt",
        "wrishp_enc",
        "wrihis_zcor",
        "classmapinterval",
    },
)

mdu_lines = set_value(mdu_lines, "TUnit", "S")
mdu_lines = set_value(mdu_lines, "TStart", "0")
mdu_lines = set_value(mdu_lines, "TStop", "2")
mdu_lines = set_value(mdu_lines, "DtUser", "0.01")
mdu_lines = set_value(mdu_lines, "DtMax", "0.01")
mdu_lines = set_value(mdu_lines, "OutputDir", str(OUTPUT_DIR))
mdu_lines = set_value(mdu_lines, "Wrihis_temperature", "0")

MDU_FILE.write_text(
    "\n".join(mdu_lines) + "\n",
    encoding="ascii",
)

# ------------------------------------------------------------
# Read hydrograph
# ------------------------------------------------------------

print("[3] Reading Scenario 02 hydrograph...")

rows = []

with HYDROGRAPH_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        if not row["Time_s"]:
            continue

        if not row["Discharge_m3s"]:
            continue

        rows.append(
            (
                float(row["Time_s"]),
                float(row["Discharge_m3s"]),
            )
        )

rows.sort(key=lambda x: x[0])

if len(rows) < 2:
    raise RuntimeError("Hydrograph has fewer than 2 rows.")

print(f"Rows: {len(rows)}")
print(f"Peak Q: {max(q for _, q in rows):.6f} m3/s")

# ------------------------------------------------------------
# Create discharge boundary
# ------------------------------------------------------------

print("[4] Creating Discharge.bc...")

bc_lines = [
    "[forcing]",
    "Name = discharge_bnd_0001",
    "Function = timeseries",
    "Time-interpolation = linear",
    "Quantity = time",
    "Unit = seconds since 1992-09-01 00:00:00",
    "Quantity = dischargebnd",
    "Unit = m3/s",
]

for t, q in rows:
    bc_lines.append(
        f"{t:.6f} {q:.9f}"
    )

# Extra zero forcing prevents EOF if the solver queries slightly later.
bc_lines.append("2.500000 0.000000000")
bc_lines.append("3.000000 0.000000000")

BC_FILE.write_text(
    "\r\n".join(bc_lines) + "\r\n",
    encoding="ascii",
)

# ------------------------------------------------------------
# Verify boundary file
# ------------------------------------------------------------

raw = BC_FILE.read_bytes()

if raw.startswith(b"\xef\xbb\xbf"):
    raise RuntimeError("Discharge.bc contains a UTF-8 BOM.")

bc_text = BC_FILE.read_text(encoding="ascii")

for required in [
    "[forcing]",
    "Name = discharge_bnd_0001",
    "Quantity = dischargebnd",
    "2.500000 0.000000000",
    "3.000000 0.000000000",
]:
    if required not in bc_text:
        raise RuntimeError(
            f"Discharge.bc validation failed: {required}"
        )

print("Boundary file: OK")

# ------------------------------------------------------------
# PROJ
# ------------------------------------------------------------

proj_db = find_proj_db()

if proj_db is None:
    raise RuntimeError("proj.db not found.")

env = os.environ.copy()
env["PROJ_DATA"] = str(proj_db.parent)
env["PROJ_LIB"] = str(proj_db.parent)

# Remove stale cache
if CACHE_FILE.exists():
    CACHE_FILE.unlink()

# ------------------------------------------------------------
# Create CMD launcher
# ------------------------------------------------------------

print("[5] Creating launcher...")

CMD_FILE.write_text(
    "@echo off\r\n"
    f'cd /d "{CASE_DIR}"\r\n'
    f'call "{RUN_BAT}" tidalflume.mdu\r\n'
    "exit /b %errorlevel%\r\n",
    encoding="ascii",
)

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

print("[6] Running D-Flow FM...\n")

process = subprocess.run(
    ["cmd.exe", "/d", "/c", CMD_FILE.name],
    cwd=str(CASE_DIR),
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

LOG_FILE.write_text(
    process.stdout,
    encoding="utf-8",
)

print(process.stdout)

# ------------------------------------------------------------
# Verify actual model result
# ------------------------------------------------------------

print("=" * 70)
print("VALIDATION")
print("=" * 70)

log = process.stdout

error_patterns = [
    r"\*\*\s*ERROR",
    r"Error found in EC-module",
    r"READING BEYOND FINAL TIME",
    r"No signals for polyline",
    r"Unknown block type",
]

errors = []

for pattern in error_patterns:

    for match in re.findall(
        pattern + r".*",
        log,
        flags=re.IGNORECASE,
    ):

        line = match.strip()

        if line not in errors:
            errors.append(line)

if process.returncode != 0:
    print(
        f"Launcher returned code {process.returncode}."
    )

    for error in errors:
        print(error)

    print(f"Log: {LOG_FILE}")
    sys.exit(1)

if errors:

    print("D-FLOW FM REPORTED ERRORS:")

    for error in errors:
        print(error)

    print(f"Log: {LOG_FILE}")
    sys.exit(1)

if "Model initialization was successful" not in log:
    print("Model initialization was not confirmed.")
    print(f"Log: {LOG_FILE}")
    sys.exit(1)

if "Computation finished" not in log:
    print("Computation was not confirmed.")
    print(f"Log: {LOG_FILE}")
    sys.exit(1)

print("SUCCESS")
print(f"Scenario              : {SCENARIO_NAME}")
print("Model initialization  : SUCCESS")
print("External forcing      : SUCCESS")
print("Computation            : SUCCESS")
print("Time unit              : SECONDS")
print("Simulation window     : 0 -> 2 s")
print(f"Output                 : {OUTPUT_DIR}")
print(f"Log                    : {LOG_FILE}")

