from pathlib import Path
import argparse
import json

import numpy as np
import torch
from neuralop.models import FNO


ROOT = Path(r"C:\JalRaksha")

DATA_DIR = ROOT / r"outputs\fno_dataset"
MODEL_DIR = ROOT / r"outputs\fno_model_physics"
OUTPUT_DIR = ROOT / r"outputs\inference"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "fno_physics_best.pt"
NORMALIZATION_FILE = DATA_DIR / "normalization.json"
MASK_FILE = DATA_DIR / "mask.npy"
X_GRID_FILE = DATA_DIR / "x_grid.npy"
Y_GRID_FILE = DATA_DIR / "y_grid.npy"


def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = FNO(
        n_modes=(8, 16),
        hidden_channels=32,
        in_channels=7,
        out_channels=1,
        n_layers=4,
    ).to(device)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, device


def normalize(value, vmin, vmax):

    if abs(vmax - vmin) < 1e-12:
        return 0.0

    return (value - vmin) / (vmax - vmin)


def predict_flood(
    reservoir_length_m,
    reservoir_height_m,
    peak_q_m3s,
    hydrograph_volume_m3,
    output_name="prediction",
):

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:
        norm = json.load(f)

    mask = np.load(
        MASK_FILE
    ).astype(np.float32)

    x_grid = np.load(
        X_GRID_FILE
    ).astype(np.float32)

    y_grid = np.load(
        Y_GRID_FILE
    ).astype(np.float32)

    Xg, Yg = np.meshgrid(
        x_grid,
        y_grid
    )

    # --------------------------------------------------------
    # Warn when inputs are outside training range
    # --------------------------------------------------------

    ranges = {
        "reservoir_length_m": (
            norm["length_min"],
            norm["length_max"],
            reservoir_length_m,
        ),
        "reservoir_height_m": (
            norm["height_min"],
            norm["height_max"],
            reservoir_height_m,
        ),
        "peak_q_m3s": (
            norm["peak_Q_min"],
            norm["peak_Q_max"],
            peak_q_m3s,
        ),
        "hydrograph_volume_m3": (
            norm["volume_min"],
            norm["volume_max"],
            hydrograph_volume_m3,
        ),
    }

    for name, (vmin, vmax, value) in ranges.items():

        if value < vmin or value > vmax:
            print(
                f"WARNING: {name}={value} is outside "
                f"training range [{vmin}, {vmax}]."
            )

    # --------------------------------------------------------
    # Construct 7 input channels
    # --------------------------------------------------------

    x_norm = (
        Xg - norm["x_min"]
    ) / (
        norm["x_max"] - norm["x_min"]
    )

    y_norm = (
        Yg - norm["y_min"]
    ) / (
        norm["y_max"] - norm["y_min"]
    )

    length_norm = normalize(
        reservoir_length_m,
        norm["length_min"],
        norm["length_max"],
    )

    height_norm = normalize(
        reservoir_height_m,
        norm["height_min"],
        norm["height_max"],
    )

    q_norm = normalize(
        peak_q_m3s,
        norm["peak_Q_min"],
        norm["peak_Q_max"],
    )

    volume_norm = normalize(
        hydrograph_volume_m3,
        norm["volume_min"],
        norm["volume_max"],
    )

    input_grid = np.stack(
        [
            x_norm,
            y_norm,
            mask,
            np.full_like(Xg, length_norm),
            np.full_like(Xg, height_norm),
            np.full_like(Xg, q_norm),
            np.full_like(Xg, volume_norm),
        ],
        axis=0,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    model, device = load_model()

    input_tensor = (
        torch.from_numpy(input_grid)
        .unsqueeze(0)
        .to(device)
    )

    mask_tensor = (
        torch.from_numpy(mask)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        raw_prediction = model(
            input_tensor
        )

        # Physics constraints:
        # non-negative depth + physical-domain mask
        prediction = (
            torch.relu(raw_prediction)
            * mask_tensor
        )

    prediction = (
        prediction
        .cpu()
        .numpy()[0, 0]
    )

    # --------------------------------------------------------
    # Convert normalized target back to metres
    # --------------------------------------------------------

    target_scale = norm["target_scale"] \
        if "target_scale" in norm \
        else None

    if target_scale is None:

        checkpoint = torch.load(
            MODEL_FILE,
            map_location="cpu",
            weights_only=False,
        )

        target_scale = float(
            checkpoint["target_scale"]
        )

    prediction_m = (
        prediction * target_scale
    ).astype(np.float32)

    # --------------------------------------------------------
    # Save prediction
    # --------------------------------------------------------

    prediction_file = (
        OUTPUT_DIR
        / f"{output_name}_depth.npy"
    )

    np.save(
        prediction_file,
        prediction_m
    )

    metadata = {
        "reservoir_length_m": reservoir_length_m,
        "reservoir_height_m": reservoir_height_m,
        "peak_q_m3s": peak_q_m3s,
        "hydrograph_volume_m3": hydrograph_volume_m3,
        "grid_shape": list(prediction_m.shape),
        "maximum_depth_m": float(prediction_m.max()),
        "mean_depth_m": float(prediction_m[mask > 0.5].mean()),
        "device": str(device),
        "model": "physics-aware FNO",
        "model_file": str(MODEL_FILE),
        "prediction_file": str(prediction_file),
    }

    metadata_file = (
        OUTPUT_DIR
        / f"{output_name}_metadata.json"
    )

    with metadata_file.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    return prediction_m, metadata


def main():

    parser = argparse.ArgumentParser(
        description="JalRaksha physics-aware FNO inference"
    )

    parser.add_argument(
        "--length",
        type=float,
        required=True,
        help="Reservoir length in metres"
    )

    parser.add_argument(
        "--height",
        type=float,
        required=True,
        help="Reservoir height in metres"
    )

    parser.add_argument(
        "--peak-q",
        type=float,
        required=True,
        help="Peak discharge in m3/s"
    )

    parser.add_argument(
        "--volume",
        type=float,
        required=True,
        help="Hydrograph volume in m3"
    )

    parser.add_argument(
        "--name",
        default="prediction",
        help="Output prediction name"
    )

    args = parser.parse_args()

    prediction, metadata = predict_flood(
        reservoir_length_m=args.length,
        reservoir_height_m=args.height,
        peak_q_m3s=args.peak_q,
        hydrograph_volume_m3=args.volume,
        output_name=args.name,
    )

    print("=" * 70)
    print("JALRAKSHA FNO INFERENCE")
    print("=" * 70)
    print("Model: physics-aware FNO")
    print("Device:", metadata["device"])
    print()
    print(
        f"Reservoir length : "
        f"{args.length:.4f} m"
    )
    print(
        f"Reservoir height : "
        f"{args.height:.4f} m"
    )
    print(
        f"Peak discharge   : "
        f"{args.peak_q:.6f} m3/s"
    )
    print(
        f"Hydrograph volume: "
        f"{args.volume:.6f} m3"
    )
    print()
    print(
        f"Prediction shape : "
        f"{prediction.shape}"
    )
    print(
        f"Maximum depth    : "
        f"{prediction.max():.6f} m"
    )
    print(
        f"Mean depth       : "
        f"{metadata['mean_depth_m']:.6f} m"
    )
    print()
    print(
        "Prediction:",
        metadata["prediction_file"]
    )
    print(
        "Metadata:",
        OUTPUT_DIR
        / f"{args.name}_metadata.json"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
