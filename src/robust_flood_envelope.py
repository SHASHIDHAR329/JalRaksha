from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(r"C:\JalRaksha")

PREDICT_SCRIPT = ROOT / "src" / "predict_flood.py"

OUTPUT_DIR = ROOT / "outputs" / "robust_flood_envelope"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORCING_SCALES = [0.975, 0.9875, 1.00, 1.025, 1.05]


def run_prediction(
    length_m: float,
    height_m: float,
    peak_q: float,
    volume_m3: float,
    name: str,
) -> np.ndarray:

    output_file = (
        ROOT
        / "outputs"
        / "inference"
        / f"{name}_depth.npy"
    )

    command = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--length",
        str(length_m),
        "--height",
        str(height_m),
        "--peak-q",
        str(peak_q),
        "--volume",
        str(volume_m3),
        "--name",
        name,
    ]

    print()
    print(
        "Running:",
        name
    )

    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    if completed.stdout:
        print(completed.stdout)

    if completed.returncode != 0:

        if completed.stderr:
            print(completed.stderr)

        raise RuntimeError(
            f"FNO inference failed for {name}"
        )

    if not output_file.exists():

        raise FileNotFoundError(
            f"Prediction file was not created:\n"
            f"{output_file}"
        )

    depth = np.load(
        output_file
    ).astype(np.float32)

    depth = np.squeeze(depth)

    if depth.ndim != 2:

        raise ValueError(
            f"Expected 2-D prediction, got {depth.shape}"
        )

    if not np.isfinite(depth).any():

        raise ValueError(
            f"Prediction contains no finite cells: {name}"
        )

    return depth


def build_envelope(
    length_m: float,
    height_m: float,
    peak_q: float,
    volume_m3: float,
    scales: list[float],
):

    results = {}

    for scale in scales:

        scaled_q = (
            peak_q * scale
        )

        scaled_volume = (
            volume_m3 * scale
        )

        tag = (
            f"{scale:.2f}x"
        ).replace(".", "_")

        name = (
            f"envelope_{tag}"
        )

        depth = run_prediction(
            length_m,
            height_m,
            scaled_q,
            scaled_volume,
            name,
        )

        depth = np.maximum(
            depth,
            0.0
        )

        results[scale] = depth

    return results


def main():

    parser = argparse.ArgumentParser(
        description=(
            "JalRaksha Robust Flood Envelope"
        )
    )

    parser.add_argument(
        "--length",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--height",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--peak-q",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--volume",
        type=float,
        required=True,
    )

    args = parser.parse_args()

    scales = FORCING_SCALES

    print("=" * 70)
    print(
        "JALRAKSHA ROBUST FLOOD ENVELOPE"
    )
    print("=" * 70)

    print(
        f"Length           : {args.length:.6f} m"
    )

    print(
        f"Height           : {args.height:.6f} m"
    )

    print(
        f"Nominal peak Q   : {args.peak_q:.6f} m3/s"
    )

    print(
        f"Nominal volume   : {args.volume:.6f} m3"
    )

    print(
        "Forcing scales   :",
        ", ".join(
            f"{s:.2f}x"
            for s in scales
        )
    )

    results = build_envelope(
        args.length,
        args.height,
        args.peak_q,
        args.volume,
        scales,
    )

    ordered = [
        results[s]
        for s in scales
    ]

    stack = np.stack(
        ordered,
        axis=0,
    )

    low = np.min(
        stack,
        axis=0,
    )

    nominal_index = scales.index(
        1.00
    )

    nominal = stack[
        nominal_index
    ]

    high = np.max(
        stack,
        axis=0,
    )

    spread = (
        high - low
    )

    robust = high

    # --------------------------------------------------------
    # Save fields
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / "lower_depth.npy",
        low,
    )

    np.save(
        OUTPUT_DIR / "nominal_depth.npy",
        nominal,
    )

    np.save(
        OUTPUT_DIR / "upper_depth.npy",
        high,
    )

    np.save(
        OUTPUT_DIR / "spread_depth.npy",
        spread,
    )

    np.save(
        OUTPUT_DIR / "robust_depth.npy",
        robust,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    nominal_max = float(
        np.max(nominal)
    )

    robust_max = float(
        np.max(robust)
    )

    mean_spread = float(
        np.mean(spread)
    )

    max_spread = float(
        np.max(spread)
    )

    material_threshold = 0.01

    material_change_cells = int(
        np.sum(
            spread >= material_threshold
        )
    )

    total_cells = int(
        spread.size
    )

    material_change_pct = (
        100.0
        * material_change_cells
        / total_cells
    )

    scenario_summary = {}

    for scale, depth in results.items():

        scenario_summary[
            f"{scale:.2f}x"
        ] = {
            "max_depth_m": float(
                np.max(depth)
            ),
            "mean_depth_m": float(
                np.mean(depth)
            ),
            "min_depth_m": float(
                np.min(depth)
            ),
        }

    summary = {

        "scenario": {
            "length_m": args.length,
            "height_m": args.height,
            "peak_q_m3s": args.peak_q,
            "volume_m3": args.volume,
        },

        "forcing_scales": scales,

        "predictions": scenario_summary,

        "envelope": {
            "nominal_max_depth_m":
                nominal_max,

            "robust_upper_max_depth_m":
                robust_max,

            "mean_spread_m":
                mean_spread,

            "maximum_spread_m":
                max_spread,

            "material_change_threshold_m":
                material_threshold,

            "material_change_cells":
                material_change_cells,

            "material_change_percent":
                material_change_pct,
        },

        "interpretation": {
            "lower_depth":
                "Minimum depth across tested forcing cases",

            "nominal_depth":
                "100 percent nominal forcing prediction",

            "upper_depth":
                "Maximum depth across tested forcing cases",

            "spread_depth":
                "Upper minus lower tested prediction",

            "robust_depth":
                "Conservative upper member of deterministic envelope",

            "warning":
                "This is sensitivity analysis, not a calibrated probability interval.",
        },
    }

    summary_file = (
        OUTPUT_DIR
        / "robust_flood_envelope.json"
    )

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ROBUST FLOOD ENVELOPE COMPLETE"
    )
    print("=" * 70)

    print(
        f"Nominal maximum depth : "
        f"{nominal_max:.6f} m"
    )

    print(
        f"Robust maximum depth  : "
        f"{robust_max:.6f} m"
    )

    print(
        f"Mean envelope spread  : "
        f"{mean_spread:.6f} m"
    )

    print(
        f"Maximum spread        : "
        f"{max_spread:.6f} m"
    )

    print(
        f"Material-change cells : "
        f"{material_change_pct:.2f}%"
    )

    print()
    print(
        f"Summary: {summary_file}"
    )

    print(
        f"Output : {OUTPUT_DIR}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The robust field is the upper member of the tested"
    )

    print(
        "forcing envelope. It is not a probability/confidence"
    )

    print(
        "interval and must not be presented as one."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
