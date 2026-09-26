from pathlib import Path
import json
import csv

import numpy as np
import torch
from neuralop.models import FNO


ROOT = Path(r"C:\JalRaksha")
DATA_DIR = ROOT / r"outputs\fno_dataset"
MODEL_DIR = ROOT / r"outputs\fno_model_masked"
BASELINE_DIR = ROOT / r"outputs\fno_model"
EVAL_DIR = MODEL_DIR / "evaluation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = np.load(DATA_DIR / "X.npy").astype(np.float32)
Y = np.load(DATA_DIR / "Y.npy").astype(np.float32)
MASK = np.load(DATA_DIR / "mask.npy").astype(np.float32)

with open(DATA_DIR / "scenario_ids.json", "r", encoding="utf-8") as f:
    scenario_ids = json.load(f)

with open(MODEL_DIR / "train_val_split.json", "r", encoding="utf-8") as f:
    split = json.load(f)

val_indices = split["validation_indices"]
target_scale = float(split["target_scale"])

mask_tensor = torch.from_numpy(MASK).to(DEVICE).unsqueeze(0).unsqueeze(0)

model = FNO(
    n_modes=(8, 16),
    hidden_channels=32,
    in_channels=7,
    out_channels=1,
    n_layers=4,
).to(DEVICE)

checkpoint = torch.load(
    MODEL_DIR / "fno_masked_best.pt",
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

X_val = torch.from_numpy(X[val_indices]).to(DEVICE)

with torch.no_grad():
    raw_pred = model(X_val)
    pred = raw_pred * mask_tensor
    pred = pred.cpu().numpy() * target_scale

true = Y[val_indices]

results = []
all_abs = []
all_sq = []

for local_i, dataset_i in enumerate(val_indices):

    scenario = scenario_ids[dataset_i]

    pred_case = pred[local_i, 0]
    true_case = true[local_i, 0]

    valid_pred = pred_case[MASK.astype(bool)]
    valid_true = true_case[MASK.astype(bool)]

    abs_error = np.abs(valid_pred - valid_true)
    sq_error = (valid_pred - valid_true) ** 2

    mae = float(abs_error.mean())
    rmse = float(np.sqrt(sq_error.mean()))

    true_max = float(valid_true.max())
    pred_max = float(valid_pred.max())

    max_error = abs(pred_max - true_max)

    outside = pred_case[MASK < 0.5]
    outside_max = float(np.max(np.abs(outside))) if len(outside) else 0.0

    negative_count = int(np.sum(valid_pred < 0))

    all_abs.extend(abs_error.tolist())
    all_sq.extend(sq_error.tolist())

    results.append({
        "scenario": scenario,
        "MAE_m": mae,
        "RMSE_m": rmse,
        "true_max_depth_m": true_max,
        "predicted_max_depth_m": pred_max,
        "max_depth_absolute_error_m": max_error,
        "outside_mask_max_depth_m": outside_max,
        "negative_prediction_cells": negative_count,
    })

    np.save(
        EVAL_DIR / f"{scenario}_prediction.npy",
        pred[local_i]
    )

    np.save(
        EVAL_DIR / f"{scenario}_truth.npy",
        true[local_i]
    )

overall_mae = float(np.mean(all_abs))
overall_rmse = float(np.sqrt(np.mean(all_sq)))

relative_errors = [
    r["max_depth_absolute_error_m"] / r["true_max_depth_m"] * 100.0
    for r in results
]

mean_relative_max_error = float(np.mean(relative_errors))

metrics_path = EVAL_DIR / "validation_metrics.csv"

with metrics_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

summary = {
    "device": str(DEVICE),
    "validation_scenarios": [scenario_ids[i] for i in val_indices],
    "overall_MAE_m": overall_mae,
    "overall_RMSE_m": overall_rmse,
    "mean_relative_max_depth_error_percent": mean_relative_max_error,
    "target_scale_m": target_scale,
    "mask_constrained": True,
}

with (EVAL_DIR / "validation_summary.json").open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("=" * 70)
print("JALRAKSHA MASK-CONSTRAINED FNO VALIDATION")
print("=" * 70)

for r in results:
    print(
        f"{r['scenario']}: "
        f"MAE={r['MAE_m']:.6f} m, "
        f"RMSE={r['RMSE_m']:.6f} m, "
        f"TrueMax={r['true_max_depth_m']:.6f} m, "
        f"PredMax={r['predicted_max_depth_m']:.6f} m, "
        f"MaxError={r['max_depth_absolute_error_m']:.6f} m, "
        f"OutsideMask={r['outside_mask_max_depth_m']:.6f} m, "
        f"NegativeCells={r['negative_prediction_cells']}"
    )

print()
print("Overall MAE:", f"{overall_mae:.6f}", "m")
print("Overall RMSE:", f"{overall_rmse:.6f}", "m")
print(
    "Mean relative max-depth error:",
    f"{mean_relative_max_error:.2f}%"
)
print()
print("Metrics:", metrics_path)
print("Predictions:", EVAL_DIR)
print("=" * 70)
