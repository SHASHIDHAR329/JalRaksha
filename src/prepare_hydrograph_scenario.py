from pathlib import Path
import csv
import re

PROJECT_ROOT = Path(r"C:\JalRaksha")
SCENARIO_NAME = "scenario_10"

FLOWTOOL_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "scenarios"
    / SCENARIO_NAME
    / "flowtool.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "sph"
    / "scenarios"
    / SCENARIO_NAME
    / "dam_break_hydrograph_clean.csv"
)

# ---------------------------------------------------------
# Read FlowTool CSV
# ---------------------------------------------------------

with FLOWTOOL_CSV.open("r", encoding="utf-8-sig") as f:
    lines = [line.strip() for line in f if line.strip()]

# Find the real header automatically
header_index = None

for i, line in enumerate(lines):
    lower = line.lower()

    if "time" in lower and "inflow" in lower:
        header_index = i
        break

if header_index is None:
    print("ERROR: Could not find the FlowTool data header.")
    print("\nFirst lines found in flowtool.csv:\n")

    for line in lines[:10]:
        print(line)

    raise SystemExit(1)

header = [x.strip() for x in lines[header_index].split(";")]

print("Detected header:")
for i, col in enumerate(header):
    print(f"  {i}: {col}")

# Find discharge column
flow_candidates = []

for i, col in enumerate(header):
    normalized = re.sub(r"[^a-z0-9]", "", col.lower())

    if "inflow" in normalized:
        flow_candidates.append((i, col))

if not flow_candidates:
    print("\nERROR: No InFlow column found.")
    raise SystemExit(1)

# Prefer Tank flow if available
flow_index = None
flow_name = None

for i, col in flow_candidates:
    normalized = re.sub(r"[^a-z0-9]", "", col.lower())

    if "tank" in normalized:
        flow_index = i
        flow_name = col
        break

# Otherwise use first available InFlow column
if flow_index is None:
    flow_index, flow_name = flow_candidates[0]

print(f"\nUsing discharge column: {flow_name}")

# ---------------------------------------------------------
# Read numerical data
# ---------------------------------------------------------

rows = []

for line in lines[header_index + 1:]:
    parts = [x.strip() for x in line.split(";")]

    if len(parts) <= flow_index:
        continue

    try:
        time_value = float(parts[0].replace(",", "."))
        flow_value = float(parts[flow_index].replace(",", "."))

        rows.append([time_value, flow_value])

    except ValueError:
        continue

if not rows:
    print("\nERROR: No numerical FlowTool data found.")
    raise SystemExit(1)

# ---------------------------------------------------------
# Clean internal zero gaps
# ---------------------------------------------------------

for i in range(1, len(rows) - 1):
    previous_q = rows[i - 1][1]
    current_q = rows[i][1]
    next_q = rows[i + 1][1]

    if current_q == 0 and previous_q > 0 and next_q > 0:
        rows[i][1] = (previous_q + next_q) / 2

# ---------------------------------------------------------
# Save clean hydrograph
# ---------------------------------------------------------

with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Time_s",
        "Discharge_m3s"
    ])

    for time_value, flow_value in rows:
        writer.writerow([
            f"{time_value:.6f}",
            f"{flow_value:.9f}"
        ])

peak_q = max(q for _, q in rows)
peak_t = next(t for t, q in rows if q == peak_q)

print("\nScenario hydrograph prepared successfully.")
print(f"Scenario: {SCENARIO_NAME}")
print(f"Rows: {len(rows)}")
print(f"Peak Q: {peak_q:.6f} m3/s")
print(f"Peak time: {peak_t:.6f} s")
print(f"Output: {OUTPUT_CSV}")
