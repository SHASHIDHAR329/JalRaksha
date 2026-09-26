from pathlib import Path
import json
import csv

import numpy as np
import pandas as pd

ROOT = Path(r"C:\JalRaksha")
DATA_DIR = ROOT / r"outputs\fno_dataset"
MODEL_DIR = ROOT / r"outputs\\fno_model_masked"
EVAL_DIR = MODEL_DIR / "evaluation"
TRUST_DIR = MODEL_DIR / "trust_engine"

TRUST_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load dataset and metadata
# ------------------------------------------------------------

X = np.load(DATA_DIR / "X.npy")
Y = np.load(DATA_DIR / "Y.npy")
MASK = np.load(DATA_DIR / "mask.npy").astype(bool)

x_grid = np.load(DATA_DIR / "x_grid.npy")
y_grid = np.load(DATA_DIR / "y_grid.npy")

with open(DATA_DIR / "scenario_ids.json", "r", encoding="utf-8") as f:
    scenario_ids = json.load(f)

with open(MODEL_DIR / "train_val_split.json", "r", encoding="utf-8") as f:
    split = json.load(f)

manifest = pd.read_csv(ROOT / r"outputs\dataset_manifest.csv")

train_indices = split["train_indices"]
val_indices = split["validation_indices"]

# ------------------------------------------------------------
# Grid spacing
# ------------------------------------------------------------

dx = float(np.mean(np.diff(x_grid)))
dy = float(np.mean(np.diff(y_grid)))
cell_area = dx * dy

# ------------------------------------------------------------
# Training envelopes
# ------------------------------------------------------------

train_truth = Y[train_indices, 0]

train_max_depths = np.array([
    float(arr[...][MASK].max()) for arr in train_truth
])

train_min_depth = float(train_max_depths.min())
train_max_depth = float(train_max_depths.max())

# Depth-gradient statistics for spatial smoothness
train_gradient_means = []

for arr in train_truth:
    gy, gx = np.gradient(arr)
    gradient = np.sqrt(gx**2 + gy**2)
    train_gradient_means.append(float(gradient[MASK].mean()))

gradient_low = float(np.percentile(train_gradient_means, 5))
gradient_high = float(np.percentile(train_gradient_means, 95))

# ------------------------------------------------------------
# Hydrograph-volume / max-depth proxy relationship
# ------------------------------------------------------------

manifest_index = manifest.set_index("scenario")

train_hydro_volume = np.array([
    float(manifest_index.loc[scenario_ids[i], "hydrograph_volume_m3"])
    for i in train_indices
])

train_depth_proxy = np.array([
    float((arr[MASK].sum()) * cell_area)
    for arr in train_truth
])

A = np.column_stack([
    train_hydro_volume,
    np.ones(len(train_hydro_volume))
])

coef, _, _, _ = np.linalg.lstsq(
    A,
    train_depth_proxy,
    rcond=None
)

proxy_slope = float(coef[0])
proxy_intercept = float(coef[1])

# ------------------------------------------------------------
# Trust evaluation
# ------------------------------------------------------------

results = []

for dataset_index in val_indices:

    scenario = scenario_ids[dataset_index]

    prediction_file = EVAL_DIR / f"{scenario}_prediction.npy"
    prediction = np.load(prediction_file)[0]

    # --------------------------------------------------------
    # Check 1: non-negative water depth
    # --------------------------------------------------------

    min_depth = float(prediction[MASK].min())

    if min_depth >= -0.001:
        nonnegative_score = 20.0
    elif min_depth >= -0.01:
        nonnegative_score = 10.0
    else:
        nonnegative_score = 0.0

    # --------------------------------------------------------
    # Check 2: outside-domain leakage
    # --------------------------------------------------------

    outside = prediction[~MASK]

    if len(outside) > 0:
        outside_max = float(np.max(np.abs(outside)))
    else:
        outside_max = 0.0

    if outside_max <= 0.001:
        mask_score = 20.0
    elif outside_max <= 0.005:
        mask_score = 10.0
    else:
        mask_score = 0.0

    # --------------------------------------------------------
    # Check 3: plausible maximum-depth envelope
    # --------------------------------------------------------

    predicted_max = float(prediction[MASK].max())

    lower_bound = train_min_depth * 0.75
    upper_bound = train_max_depth * 1.25

    if lower_bound <= predicted_max <= upper_bound:
        range_score = 20.0
    elif train_min_depth * 0.50 <= predicted_max <= train_max_depth * 1.50:
        range_score = 10.0
    else:
        range_score = 0.0

    # --------------------------------------------------------
    # Check 4: hydrograph/depth-volume consistency
    #
    # NOTE:
    # This is a proxy because Y is maximum depth, not an
    # instantaneous water-volume field.
    # --------------------------------------------------------

    hydro_volume = float(
        manifest_index.loc[scenario, "hydrograph_volume_m3"]
    )

    predicted_proxy = float(prediction[MASK].sum() * cell_area)

    expected_proxy = proxy_slope * hydro_volume + proxy_intercept

    if expected_proxy > 0:
        proxy_ratio = predicted_proxy / expected_proxy
    else:
        proxy_ratio = float("inf")

    if 0.70 <= proxy_ratio <= 1.30:
        proxy_score = 20.0
    elif 0.50 <= proxy_ratio <= 1.50:
        proxy_score = 10.0
    else:
        proxy_score = 0.0

    # --------------------------------------------------------
    # Check 5: spatial smoothness
    # --------------------------------------------------------

    gy, gx = np.gradient(prediction)
    gradient = np.sqrt(gx**2 + gy**2)
    gradient_mean = float(gradient[MASK].mean())

    if gradient_low <= gradient_mean <= gradient_high:
        smoothness_score = 20.0
    elif (
        gradient_low * 0.5 <= gradient_mean <= gradient_high * 2.0
    ):
        smoothness_score = 10.0
    else:
        smoothness_score = 0.0

    trust_score = (
        nonnegative_score
        + mask_score
        + range_score
        + proxy_score
        + smoothness_score
    )

    if trust_score >= 80:
        status = "TRUSTED"
    elif trust_score >= 60:
        status = "REVIEW"
    else:
        status = "REJECT"

    results.append({
        "scenario": scenario,
        "trust_score_100": trust_score,
        "status": status,
        "min_depth_m": min_depth,
        "outside_mask_max_depth_m": outside_max,
        "predicted_max_depth_m": predicted_max,
        "train_max_depth_min_m": train_min_depth,
        "train_max_depth_max_m": train_max_depth,
        "hydrograph_volume_m3": hydro_volume,
        "depth_volume_proxy_m3": predicted_proxy,
        "expected_proxy_m3": expected_proxy,
        "proxy_ratio": proxy_ratio,
        "gradient_mean": gradient_mean,
    })

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

report_path = TRUST_DIR / "trust_report.csv"

with report_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

summary = {
    "validation_scenarios": [r["scenario"] for r in results],
    "scores_are_prototype_sanity_checks": True,
    "important_note": (
        "Depth-volume consistency is a proxy because the FNO target "
        "is maximum flood depth rather than instantaneous water volume."
    ),
    "grid_dx_m": dx,
    "grid_dy_m": dy,
    "training_max_depth_min_m": train_min_depth,
    "training_max_depth_max_m": train_max_depth,
    "gradient_training_p05": gradient_low,
    "gradient_training_p95": gradient_high,
    "proxy_slope": proxy_slope,
    "proxy_intercept": proxy_intercept,
    "results": results,
}

with (TRUST_DIR / "trust_summary.json").open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("=" * 70)
print("JALRAKSHA MASK-CONSTRAINED FNO TRUST ENGINE")
print("=" * 70)

for r in results:
    print(
        f"{r['scenario']}: "
        f"Score={r['trust_score_100']:.1f}/100 | "
        f"Status={r['status']} | "
        f"MaxDepth={r['predicted_max_depth_m']:.6f} m | "
        f"ProxyRatio={r['proxy_ratio']:.3f}"
    )

print()
print("Report :", report_path)
print("Summary:", TRUST_DIR / "trust_summary.json")
print()
print("NOTE: Trust score is a prototype engineering screening layer,")
print("not a certified flood-model confidence probability.")
print("=" * 70)

