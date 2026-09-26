from pathlib import Path
import pandas as pd
import json

ROOT = Path(r"C:\JalRaksha")

BASELINE_EVAL = ROOT / r"outputs\fno_model\evaluation"
PHYSICS_EVAL = ROOT / r"outputs\fno_model_physics\evaluation"

BASELINE_TRUST = ROOT / r"outputs\fno_model\trust_engine\trust_report.csv"
PHYSICS_TRUST = ROOT / r"outputs\fno_model_physics\trust_engine\trust_report.csv"

OUT_DIR = ROOT / r"outputs\fno_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load validation metrics
# ------------------------------------------------------------

baseline = pd.read_csv(
    BASELINE_EVAL / "validation_metrics.csv"
)

physics = pd.read_csv(
    PHYSICS_EVAL / "validation_metrics.csv"
)

# ------------------------------------------------------------
# Load trust reports for physical checks
# ------------------------------------------------------------

baseline_trust = pd.read_csv(BASELINE_TRUST)
physics_trust = pd.read_csv(PHYSICS_TRUST)

baseline_outside_mask = float(
    baseline_trust["outside_mask_max_depth_m"].max()
)

physics_outside_mask = float(
    physics_trust["outside_mask_max_depth_m"].max()
)

# ------------------------------------------------------------
# Calculate metrics
# ------------------------------------------------------------

baseline_mae = float(baseline["MAE_m"].mean())
physics_mae = float(physics["MAE_m"].mean())

baseline_rmse = float(baseline["RMSE_m"].mean())
physics_rmse = float(physics["RMSE_m"].mean())

baseline_max_error = float(
    (
        baseline["max_depth_absolute_error_m"]
        / baseline["true_max_depth_m"]
        * 100
    ).mean()
)

physics_max_error = float(
    (
        physics["max_depth_absolute_error_m"]
        / physics["true_max_depth_m"]
        * 100
    ).mean()
)

mae_improvement = (
    1 - physics_mae / baseline_mae
) * 100

rmse_improvement = (
    1 - physics_rmse / baseline_rmse
) * 100

max_depth_improvement = (
    1 - physics_max_error / baseline_max_error
) * 100

# ------------------------------------------------------------
# Build comparison
# ------------------------------------------------------------

comparison = {
    "baseline_fno": {
        "MAE_m": baseline_mae,
        "RMSE_m": baseline_rmse,
        "mean_relative_max_depth_error_percent": baseline_max_error,
        "outside_mask_max_depth_m": baseline_outside_mask,
    },
    "physics_aware_fno": {
        "MAE_m": physics_mae,
        "RMSE_m": physics_rmse,
        "mean_relative_max_depth_error_percent": physics_max_error,
        "outside_mask_max_depth_m": physics_outside_mask,
    },
    "improvement": {
        "MAE_percent": mae_improvement,
        "RMSE_percent": rmse_improvement,
        "max_depth_error_percent": max_depth_improvement,
    },
}

# ------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------

json_path = OUT_DIR / "AI_MODEL_COMPARISON.json"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

rows = [
    {
        "model": "Baseline FNO",
        "MAE_m": baseline_mae,
        "RMSE_m": baseline_rmse,
        "mean_relative_max_depth_error_percent": baseline_max_error,
        "outside_mask_max_depth_m": baseline_outside_mask,
    },
    {
        "model": "Physics-aware FNO",
        "MAE_m": physics_mae,
        "RMSE_m": physics_rmse,
        "mean_relative_max_depth_error_percent": physics_max_error,
        "outside_mask_max_depth_m": physics_outside_mask,
    },
]

csv_path = OUT_DIR / "AI_MODEL_COMPARISON.csv"

pd.DataFrame(rows).to_csv(csv_path, index=False)

# ------------------------------------------------------------
# Print report
# ------------------------------------------------------------

print("=" * 70)
print("JALRAKSHA AI MODEL COMPARISON")
print("=" * 70)

print(
    f"Baseline FNO MAE:              {baseline_mae:.6f} m"
)

print(
    f"Physics-aware FNO MAE:         {physics_mae:.6f} m"
)

print(
    f"MAE improvement:               {mae_improvement:.2f}%"
)

print()

print(
    f"Baseline FNO RMSE:             {baseline_rmse:.6f} m"
)

print(
    f"Physics-aware FNO RMSE:        {physics_rmse:.6f} m"
)

print(
    f"RMSE improvement:              {rmse_improvement:.2f}%"
)

print()

print(
    f"Baseline max-depth error:      {baseline_max_error:.2f}%"
)

print(
    f"Physics-aware max-depth error: {physics_max_error:.2f}%"
)

print(
    f"Max-depth improvement:         {max_depth_improvement:.2f}%"
)

print()

print(
    f"Baseline outside-mask:         {baseline_outside_mask:.6f} m"
)

print(
    f"Physics-aware outside-mask:    {physics_outside_mask:.6f} m"
)

print()

print("CSV :", csv_path)
print("JSON:", json_path)

print("=" * 70)
