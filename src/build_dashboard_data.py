from pathlib import Path
import json
import pandas as pd


ROOT = Path(r"C:\JalRaksha")

OUT_DIR = ROOT / r"outputs\dashboard_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

physics_eval = pd.read_csv(
    ROOT / r"outputs\fno_model_physics\evaluation\validation_metrics.csv"
)

trust = pd.read_csv(
    ROOT / r"outputs\fno_model_physics\trust_engine\trust_report.csv"
)

impact = pd.read_csv(
    ROOT / r"outputs\impact_analysis\flood_impact_summary.csv"
)

comparison_path = ROOT / r"outputs\fno_model\AI_MODEL_COMPARISON.json"

with comparison_path.open("r", encoding="utf-8") as f:
    comparison_data = json.load(f)


scenarios = []

for _, ev in physics_eval.iterrows():

    scenario = ev["scenario"]

    trust_row = trust[
        trust["scenario"] == scenario
    ].iloc[0]

    impact_row = impact[
        impact["scenario"] == scenario
    ].iloc[0]

    scenarios.append({
        "scenario": scenario,

        "fno": {
            "mae_m": float(ev["MAE_m"]),
            "rmse_m": float(ev["RMSE_m"]),
            "true_max_depth_m": float(
                ev["true_max_depth_m"]
            ),
            "predicted_max_depth_m": float(
                ev["predicted_max_depth_m"]
            ),
            "max_depth_absolute_error_m": float(
                ev["max_depth_absolute_error_m"]
            ),
        },

        "trust": {
            "score": float(
                trust_row["trust_score_100"]
            ),
            "status": str(
                trust_row["status"]
            ),
            "outside_mask_max_depth_m": float(
                trust_row["outside_mask_max_depth_m"]
            ),
            "proxy_ratio": float(
                trust_row["proxy_ratio"]
            ),
        },

        "impact": {
            "maximum_depth_m": float(
                impact_row["maximum_depth_m"]
            ),
            "mean_depth_m": float(
                impact_row["mean_depth_m"]
            ),
            "flooded_area_gt_0p01m_m2": float(
                impact_row["flooded_area_gt_0p01m"]
            ),
            "high_severity_area_ge_0p20m_m2": float(
                impact_row["high_severity_area_ge_0p20m"]
            ),
        }
    })


dashboard = {
    "project": "JalRaksha",

    "model": {
        "type": "Physics-aware FNO",
        "input_shape": [7, 16, 128],
        "output_shape": [1, 16, 128],
        "validation_scenarios": [
            s["scenario"] for s in scenarios
        ],
    },

    "performance": {
        "baseline": comparison_data["baseline_fno"],
        "physics_aware": comparison_data["physics_aware_fno"],
        "improvement": comparison_data["improvement"],
    },

    "scenarios": scenarios,

    "notes": [
        "FNO metrics are prototype validation results.",
        "Trust score is an engineering screening layer, not a certified probability.",
        "Impact classes are prototype screening classes.",
        "Current flume coordinates are laboratory coordinates, not geographic satellite coordinates."
    ]
}


out_path = OUT_DIR / "jalraksha_dashboard.json"

with out_path.open("w", encoding="utf-8") as f:
    json.dump(dashboard, f, indent=2)


print("=" * 70)
print("JALRAKSHA DASHBOARD DATA PACKAGE")
print("=" * 70)

print("Scenarios:", len(scenarios))

print(
    "Physics-aware MAE:",
    f"{dashboard['performance']['physics_aware']['MAE_m']:.6f} m"
)

print(
    "Physics-aware RMSE:",
    f"{dashboard['performance']['physics_aware']['RMSE_m']:.6f} m"
)

print(
    "MAE improvement:",
    f"{dashboard['performance']['improvement']['MAE_percent']:.2f}%"
)

for s in scenarios:
    print(
        f"{s['scenario']}: "
        f"Trust={s['trust']['score']:.0f}/100 "
        f"{s['trust']['status']} | "
        f"MaxDepth={s['impact']['maximum_depth_m']:.6f} m | "
        f"FloodedArea={s['impact']['flooded_area_gt_0p01m_m2']:.2f} m²"
    )

print()
print("Output:", out_path)
print("=" * 70)
