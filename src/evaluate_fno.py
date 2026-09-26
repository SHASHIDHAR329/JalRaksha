from pathlib import Path
import json
import csv

import numpy as np
import torch
from neuralop.models import FNO


ROOT = Path(r"C:\JalRaksha")
DATA_DIR = ROOT / r"outputs\fno_dataset"
MODEL_DIR = ROOT / r"outputs\fno_model"
EVAL_DIR = MODEL_DIR / "evaluation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

X = np.load(DATA_DIR / "X.npy").astype(np.float32)
Y = np.load(DATA_DIR / "Y.npy").astype(np.float32)
MASK = np.load(DATA_DIR / "mask.npy").astype(np.float32)

with open(DATA_DIR / "scenario_ids.json", "r", encoding="utf-8") as f:
    scenario_ids = json.load(f)

with open(MODEL_DIR / "train_val_split.json", "r", encoding="utf-8") as f:
    split = json.load(f)

with open(DATA_DIR / "normalization.json", "r", encoding="utf-8") as f:
    normalization = json.load(f)

val_indices = split["validation_indices"]
target_scale = float(split["target_scale"])

# ------------------------------------------------------------
# Rebuild identical model
# ------------------------------------------------------------

model = FNO(
    n_modes=(8, 16),
    hidden_channels=32,
    in_channels=7,
    out_channels=1,
    n_layers=4,
).to(DEVICE)

checkpoint = torch.load(
    MODEL_DIR / "fno_baseline_best.pt",
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ------------------------------------------------------------
# Predict validation scenarios
# ------------------------------------------------------------

X_val = torch.from_numpy(X[val_indices]).to(DEVICE)

with torch.no_grad():
    pred_scaled = model(X_val).cpu().numpy()

pred = pred_scaled * target_scale

true = Y[val_indices]

# ------------------------------------------------------------
# Apply common physical mask
# ------------------------------------------------------------

mask = MASK.astype(bool)

results = []

all_abs = []
all_sq = []

for local_i, dataset_i in enumerate(val_indices):

    scenario = scenario_ids[dataset_i]

    pred_case = pred[local_i, 0]
    true_case = true[local_i, 0]

    valid_pred = pred_case[mask]
    valid_true = true_case[mask]

    abs_error = np.abs(valid_pred - valid_true)
    sq_error = (valid_pred - valid_true) ** 2

    mae = float(abs_error.mean())
    rmse = float(np.sqrt(sq_error.mean()))

    true_max = float(valid_true.max())
    pred_max = float(valid_pred.max())

    max_depth_error = abs(pred_max - true_max)

    all_abs.extend(abs_error.tolist())
    all_sq.extend(sq_error.tolist())

    results.append({
        "scenario": scenario,
        "MAE_m": mae,
        "RMSE_m": rmse,
        "true_max_depth_m": true_max,
        "predicted_max_depth_m": pred_max,
        "max_depth_absolute_error_m": max_depth_error,
    })

    np.save(
        EVAL_DIR / f"{scenario}_prediction.npy",
        pred[local_i]
    )

    np.save(
        EVAL_DIR / f"{scenario}_truth.npy",
        true[local_i]
    )

# ------------------------------------------------------------
# Overall metrics
# ------------------------------------------------------------

overall_mae = float(np.mean(all_abs))
overall_rmse = float(np.sqrt(np.mean(all_sq)))

# Relative errors against true maximum depth
relative_max_errors = []

for r in results:
    true_max = r["true_max_depth_m"]

    if true_max > 0:
        relative_max_errors.append(
            r["max_depth_absolute_error_m"] / true_max * 100.0
        )

mean_relative_max_error = float(np.mean(relative_max_errors))

# ------------------------------------------------------------
# Save metrics
# ------------------------------------------------------------

metrics_path = EVAL_DIR / "validation_metrics.csv"

with metrics_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "scenario",
            "MAE_m",
            "RMSE_m",
            "true_max_depth_m",
            "predicted_max_depth_m",
            "max_depth_absolute_error_m",
        ],
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
}

with (EVAL_DIR / "validation_summary.json").open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

# ------------------------------------------------------------
# Print report
# ------------------------------------------------------------

print("=" * 70)
print("JALRAKSHA FNO VALIDATION")
print("=" * 70)

print("Device:", DEVICE)
print()
print("Validation scenarios:")

for r in results:
    print(
        f"  {r['scenario']}: "
        f"MAE={r['MAE_m']:.6f} m, "
        f"RMSE={r['RMSE_m']:.6f} m, "
        f"TrueMax={r['true_max_depth_m']:.6f} m, "
        f"PredMax={r['predicted_max_depth_m']:.6f} m, "
        f"MaxError={r['max_depth_absolute_error_m']:.6f} m"
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
print("Summary:", EVAL_DIR / "validation_summary.json")
print("Predictions:", EVAL_DIR)
print("=" * 70)
