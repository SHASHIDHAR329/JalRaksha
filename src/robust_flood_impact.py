from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_array(folder: Path, name: str) -> np.ndarray:
    path = folder / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path).astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-folder", required=True)
    parser.add_argument(
        "--mask",
        default=r"C:\JalRaksha\outputs\fno_dataset\mask.npy"
    )
    args = parser.parse_args()

    folder = Path(args.scenario_folder)
    mask_path = Path(args.mask)

    lower = load_array(folder, "lower_depth.npy")
    nominal = load_array(folder, "nominal_depth.npy")
    upper = load_array(folder, "upper_depth.npy")
    spread = load_array(folder, "spread_depth.npy")
    robust = load_array(folder, "robust_depth.npy")
    mask = np.load(mask_path).astype(bool)

    if nominal.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: nominal={nominal.shape}, mask={mask.shape}"
        )

    valid = mask & np.isfinite(nominal) & np.isfinite(robust)

    if not np.any(valid):
        raise ValueError("No valid model cells found.")

    nominal_v = nominal[valid]
    robust_v = robust[valid]
    spread_v = spread[valid]

    change_threshold_m = 0.0005
    changed = valid & (spread > change_threshold_m)

    result = {
        "scenario_folder": str(folder),
        "grid_shape": list(nominal.shape),
        "valid_cells": int(valid.sum()),

        "nominal": {
            "max_depth_m": float(np.max(nominal_v)),
            "mean_depth_m": float(np.mean(nominal_v)),
            "p95_depth_m": float(np.percentile(nominal_v, 95))
        },

        "robust": {
            "max_depth_m": float(np.max(robust_v)),
            "mean_depth_m": float(np.mean(robust_v)),
            "p95_depth_m": float(np.percentile(robust_v, 95))
        },

        "envelope": {
            "mean_spread_m": float(np.mean(spread_v)),
            "p95_spread_m": float(np.percentile(spread_v, 95)),
            "max_spread_m": float(np.max(spread_v)),
            "max_depth_lift_m": float(
                np.max(robust_v) - np.max(nominal_v)
            ),
            "changed_cells_above_0_5mm": int(changed.sum()),
            "changed_cell_fraction_pct": float(
                100.0 * changed.sum() / valid.sum()
            )
        },

        "interpretation": (
            "Deterministic sensitivity envelope from the tested "
            "forcing cases. It is not a probability or confidence interval."
        )
    }

    output_dir = Path(r"C:\JalRaksha\outputs\robust_flood_impact")
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_name = folder.name.replace(
        "robust_flood_envelope_", ""
    )

    output_file = output_dir / f"{scenario_name}_impact.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 70)
    print("JALRAKSHA ROBUST FLOOD IMPACT ANALYSIS")
    print("=" * 70)
    print(f"Scenario folder       : {folder}")
    print(f"Valid cells            : {result['valid_cells']}")
    print(
        f"Nominal max depth     : "
        f"{result['nominal']['max_depth_m']:.6f} m"
    )
    print(
        f"Robust max depth      : "
        f"{result['robust']['max_depth_m']:.6f} m"
    )
    print(
        f"Mean envelope spread  : "
        f"{result['envelope']['mean_spread_m']:.6f} m"
    )
    print(
        f"P95 envelope spread   : "
        f"{result['envelope']['p95_spread_m']:.6f} m"
    )
    print(
        f"Maximum spread        : "
        f"{result['envelope']['max_spread_m']:.6f} m"
    )
    print(
        f"Maximum depth lift    : "
        f"{result['envelope']['max_depth_lift_m']:.6f} m"
    )
    print(
        f"Changed cells >0.5mm  : "
        f"{result['envelope']['changed_cell_fraction_pct']:.2f}%"
    )
    print()
    print(f"Report: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
