from pathlib import Path
import csv
import os
import re
import shutil
import subprocess
import sys


# ============================================================
# JALRAKSHA FLUME - FINAL CLEAN D-FLOW FM RUN
# ============================================================

PROJECT_ROOT = Path(r"C:\JalRaksha")

CASE_DIR = (
    PROJECT_ROOT
    / "simulations"
    / "dflowfm"
    / "jalraksha_flume"
)

MDU_FILE = CASE_DIR / "tidalflume.mdu"
BC_FILE = CASE_DIR / "Discharge.bc"
CACHE_FILE = CASE_DIR / "tidalflume.cache"

HYDROGRAPH_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "scenarios"
    / "scenario_01"
    / "dam_break_hydrograph_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dflowfm"
    / "jalraksha_flume"
)

RUN_BAT = Path(
    r"C:\Program Files\Deltares\Delft3D FM Suite 2026.01 HMWQ"
) / "plugins" / "DeltaShell.Dimr" / "kernels" / "x64" / "bin" / "run_dflowfm.bat"

CMD_FILE = CASE_DIR / "run_jalraksha_flume.cmd"
LOG_FILE = CASE_DIR / "jalraksha_flume_run.log"


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def set_mdu_value(
    lines: list[str],
    keyword: str,
    value: str,
) -> list[str]:
    pattern = re.compile(
        rf"^\s*{re.escape(keyword)}\s*=",
        re.IGNORECASE,
    )

    result = []
    found = False

    for line in lines:
        if pattern.match(line):
            result.append(f"{keyword} = {value}")
            found = True
        else:
            result.append(line)

    if not found:
        result.append(f"{keyword} = {value}")

    return result


def remove_mdu_keys(
    lines: list[str],
    keys: set[str],
) -> list[str]:

    lower_keys = {key.lower() for key in keys}
    result = []

    for line in lines:

        if "=" not in line:
            result.append(line)
            continue

        key = line.split("=", 1)[0].strip().lower()

        if key in lower_keys:
            continue

        result.append(line)

    return result


def find_proj_db() -> Path | None:

    root = Path(
        r"C:\Program Files\Deltares\Delft3D FM Suite 2026.01 HMWQ"
    )

    if not root.exists():
        return None

    matches = list(root.rglob("proj.db"))

    if not matches:
        return None

    return matches[0]


# ============================================================
# CHECK INPUTS
# ============================================================

print("=" * 70)
print("JALRAKSHA FLUME - FINAL CLEAN D-FLOW FM RUN")
print("=" * 70)

require_file(CASE_DIR, "D-Flow FM case directory")
require_file(MDU_FILE, "D-Flow FM MDU file")
require_file(HYDROGRAPH_FILE, "SPH hydrograph")
require_file(RUN_BAT, "D-Flow FM launcher")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BACKUPS
# ============================================================

print("\n[1/8] Creating backups...")

shutil.copy2(
    MDU_FILE,
    CASE_DIR / "tidalflume.mdu.before_final_timeunit_fix.bak",
)

if BC_FILE.exists():
    shutil.copy2(
        BC_FILE,
        CASE_DIR / "Discharge.bc.before_final_timeunit_fix.bak",
    )

print("Backups created.")


# ============================================================
# CLEAN MDU
# ============================================================

print("\n[2/8] Cleaning MDU and fixing time units...")

mdu_lines = MDU_FILE.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines()

# Remove obsolete/unused settings.
mdu_lines = remove_mdu_keys(
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

# IMPORTANT:
# D-Flow FM must interpret TStart/TStop in SECONDS.
mdu_lines = set_mdu_value(
    mdu_lines,
    "TUnit",
    "S",
)

mdu_lines = set_mdu_value(
    mdu_lines,
    "TStart",
    "0",
)

mdu_lines = set_mdu_value(
    mdu_lines,
    "TStop",
    "2",
)

mdu_lines = set_mdu_value(
    mdu_lines,
    "DtUser",
    "0.01",
)

mdu_lines = set_mdu_value(
    mdu_lines,
    "DtMax",
    "0.01",
)

mdu_lines = set_mdu_value(
    mdu_lines,
    "OutputDir",
    str(OUTPUT_DIR),
)

mdu_lines = set_mdu_value(
    mdu_lines,
    "Wrihis_temperature",
    "0",
)

# Force ASCII so no BOM is introduced.
MDU_FILE.write_text(
    "\n".join(mdu_lines) + "\n",
    encoding="ascii",
)

print("MDU fixed.")


# ============================================================
# READ HYDROGRAPH
# ============================================================

print("\n[3/8] Reading SPH hydrograph...")

rows = []

with HYDROGRAPH_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    required = {
        "Time_s",
        "Discharge_m3s",
    }

    if not reader.fieldnames:
        raise RuntimeError(
            "Hydrograph header is missing."
        )

    if not required.issubset(
        set(reader.fieldnames)
    ):
        raise RuntimeError(
            "Hydrograph must contain Time_s and Discharge_m3s."
        )

    for record in reader:

        if not record["Time_s"]:
            continue

        if not record["Discharge_m3s"]:
            continue

        t = float(record["Time_s"])
        q = float(record["Discharge_m3s"])

        rows.append((t, q))

rows.sort(key=lambda item: item[0])

if len(rows) < 2:
    raise RuntimeError(
        "Hydrograph contains fewer than 2 data points."
    )

print(f"Hydrograph rows: {len(rows)}")
print(f"Hydrograph end : {rows[-1][0]:.6f} s")


# ============================================================
# CREATE D-FLOW FM BOUNDARY FILE
# ============================================================

print("\n[4/8] Creating clean Discharge.bc...")

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

# The model is exactly 2 s, but the solver can query slightly beyond
# the nominal endpoint. Keep valid zero forcing beyond the end.
bc_lines.append(
    "2.500000 0.000000000"
)

bc_lines.append(
    "3.000000 0.000000000"
)

# ASCII = NO UTF-8 BOM
with BC_FILE.open(
    "w",
    encoding="ascii",
    newline="\r\n",
) as f:

    f.write(
        "\r\n".join(bc_lines)
        + "\r\n"
    )

print("Discharge.bc created.")
print("Signal: discharge_bnd_0001")
print("Encoding: ASCII / no BOM")


# ============================================================
# VALIDATE BC
# ============================================================

print("\n[5/8] Validating boundary file...")

raw = BC_FILE.read_bytes()

if raw.startswith(b"\xef\xbb\xbf"):
    raise RuntimeError(
        "Discharge.bc contains a UTF-8 BOM."
    )

bc_text = BC_FILE.read_text(
    encoding="ascii"
)

required_strings = [
    "[forcing]",
    "Name = discharge_bnd_0001",
    "Function = timeseries",
    "Quantity = time",
    "Quantity = dischargebnd",
    "Unit = m3/s",
    "2.500000 0.000000000",
    "3.000000 0.000000000",
]

for item in required_strings:

    if item not in bc_text:
        raise RuntimeError(
            f"Discharge.bc validation failed: {item}"
        )

print("Boundary file validation: SUCCESS")


# ============================================================
# PROJ ENVIRONMENT
# ============================================================

print("\n[6/8] Configuring PROJ...")

proj_db = find_proj_db()

if proj_db is None:
    raise RuntimeError(
        "Could not find proj.db."
    )

env = os.environ.copy()

env["PROJ_DATA"] = str(proj_db.parent)
env["PROJ_LIB"] = str(proj_db.parent)

print(f"proj.db: {proj_db}")

# Remove stale D-Flow cache.
if CACHE_FILE.exists():
    CACHE_FILE.unlink()

print("Old cache removed.")


# ============================================================
# CREATE TEMP CMD LAUNCHER
# ============================================================

print("\n[7/8] Creating launcher...")

cmd_contents = (
    "@echo off\r\n"
    f'cd /d "{CASE_DIR}"\r\n'
    f'call "{RUN_BAT}" tidalflume.mdu\r\n'
    "exit /b %errorlevel%\r\n"
)

CMD_FILE.write_text(
    cmd_contents,
    encoding="ascii",
)

print(f"Launcher: {CMD_FILE}")


# ============================================================
# RUN D-FLOW FM
# ============================================================

print("\n[8/8] Running D-Flow FM...")
print()

process = subprocess.run(
    [
        "cmd.exe",
        "/d",
        "/c",
        CMD_FILE.name,
    ],
    cwd=str(CASE_DIR),
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

log = process.stdout

LOG_FILE.write_text(
    log,
    encoding="utf-8",
)

print(log)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("=" * 70)
print("FINAL D-FLOW FM VALIDATION")
print("=" * 70)

error_patterns = [
    r"\*\*\s*ERROR",
    r"Error found in EC-module",
    r"READING BEYOND FINAL TIME",
    r"No signals for polyline",
    r"Unknown block type",
]

errors = []

for pattern in error_patterns:

    matches = re.findall(
        pattern + r".*",
        log,
        flags=re.IGNORECASE,
    )

    for match in matches:

        line = match.strip()

        if line not in errors:
            errors.append(line)


if process.returncode != 0:

    print(
        f"D-Flow launcher returned code {process.returncode}."
    )

    for error in errors:
        print(error)

    print(f"\nFull log: {LOG_FILE}")

    sys.exit(1)


if errors:

    print("D-FLOW FM REPORTED ERRORS:")

    for error in errors:
        print(error)

    print(f"\nFull log: {LOG_FILE}")

    sys.exit(1)


if "Model initialization was successful" not in log:

    print(
        "Model initialization success marker was not found."
    )

    print(f"\nFull log: {LOG_FILE}")

    sys.exit(1)


if "Computation finished" not in log:

    print(
        "Computation finished marker was not found."
    )

    print(f"\nFull log: {LOG_FILE}")

    sys.exit(1)


# ============================================================
# SUCCESS
# ============================================================

print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)

print("NO D-FLOW FM MODEL ERROR FOUND.")
print("Model initialization : SUCCESS")
print("External forcing     : SUCCESS")
print("Computation           : SUCCESS")
print("Time unit             : SECONDS")
print("Simulation window     : 0 -> 2 s")
print(f"Output directory      : {OUTPUT_DIR}")
print(f"Run log               : {LOG_FILE}")