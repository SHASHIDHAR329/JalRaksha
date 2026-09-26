from pathlib import Path
import csv

ROOT = Path(r"C:\JalRaksha")

scenario_params = {
    "scenario_01": (1.00, 2.00),
    "scenario_02": (1.50, 2.00),
    "scenario_03": (2.00, 2.00),
    "scenario_04": (1.25, 1.50),
    "scenario_05": (1.50, 1.50),
    "scenario_06": (1.75, 1.50),
    "scenario_07": (2.00, 1.50),
    "scenario_08": (1.25, 1.75),
    "scenario_09": (1.50, 1.75),
    "scenario_10": (1.75, 1.75),
}

def hydro_path(name):
    return ROOT / "outputs" / "sph" / "scenarios" / name / "dam_break_hydrograph_clean.csv"

def flood_path(name):
    if name == "scenario_01":
        return ROOT / "outputs" / "dflowfm" / "jalraksha_flume" / "flood_depth_cells.csv"
    return ROOT / "outputs" / "dflowfm" / f"jalraksha_flume_{name}" / "flood_depth_cells.csv"

manifest = []

for name, (length, height) in scenario_params.items():
    hp = hydro_path(name)
    fp = flood_path(name)

    peak_q = None
    volume = None
    max_depth = None
    mean_depth = None

    with hp.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    times = []
    flows = []

    for row in rows:
        try:
            times.append(float(row["Time_s"]))
            flows.append(float(row["Discharge_m3s"]))
        except (KeyError, ValueError):
            pass

    if flows:
        peak_q = max(flows)

    if len(times) > 1:
        volume = sum(
            0.5 * (flows[i] + flows[i - 1]) * (times[i] - times[i - 1])
            for i in range(1, len(times))
        )

    with fp.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    depth_column = next(
        col for col in rows[0].keys()
        if "depth" in col.lower()
    )

    depths = []
    for row in rows:
        try:
            depths.append(float(row[depth_column]))
        except (ValueError, TypeError):
            pass

    if depths:
        max_depth = max(depths)
        mean_depth = sum(depths) / len(depths)

    manifest.append({
        "scenario": name,
        "reservoir_length_m": length,
        "reservoir_height_m": height,
        "peak_Q_m3s": peak_q,
        "hydrograph_volume_m3": volume,
        "max_depth_m": max_depth,
        "mean_depth_m": mean_depth,
    })

out = ROOT / "outputs" / "dataset_manifest.csv"

with out.open("w", encoding="utf-8", newline="") as f:
    fields = list(manifest[0].keys())
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(manifest)

print("=" * 70)
print("DATASET MANIFEST REPAIRED")
print("=" * 70)

for r in manifest:
    print(
        f"{r['scenario']}: "
        f"L={r['reservoir_length_m']:.2f} m, "
        f"H={r['reservoir_height_m']:.2f} m, "
        f"PeakQ={r['peak_Q_m3s']:.6f}, "
        f"Volume={r['hydrograph_volume_m3']:.6f}, "
        f"MaxDepth={r['max_depth_m']:.6f}"
    )

print(f"\nManifest: {out}")
