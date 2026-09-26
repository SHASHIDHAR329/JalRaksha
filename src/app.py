from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from predict_flood import predict_flood


ROOT = Path(r"C:\JalRaksha")

DATA_DIR = ROOT / r"outputs\fno_dataset"
MODEL_DIR = ROOT / r"outputs\fno_model_physics"
DASHBOARD_FILE = ROOT / r"outputs\dashboard_data\jalraksha_dashboard.json"
NORM_FILE = DATA_DIR / "normalization.json"
MASK_FILE = DATA_DIR / "mask.npy"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="JalRaksha",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# LOAD DATA
# ============================================================

with DASHBOARD_FILE.open("r", encoding="utf-8") as f:
    dashboard = json.load(f)

with NORM_FILE.open("r", encoding="utf-8") as f:
    normalization = json.load(f)

mask = np.load(MASK_FILE).astype(bool)

performance = dashboard["performance"]
validation_scenarios = dashboard["scenarios"]


# ============================================================
# HEADER
# ============================================================

st.title("🌊 JalRaksha")

st.subheader(
    "Physics-Aware AI Flood Prediction Prototype"
)

st.caption(
    "SPH → Hydrograph → D-Flow FM → Physics-aware FNO "
    "→ Trust Engine → Impact Analysis"
)


# ============================================================
# TOP PERFORMANCE METRICS
# ============================================================

st.markdown("## AI Performance")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "FNO MAE",
        f"{performance['physics_aware']['MAE_m']:.6f} m"
    )

with c2:
    st.metric(
        "FNO RMSE",
        f"{performance['physics_aware']['RMSE_m']:.6f} m"
    )

with c3:
    st.metric(
        "Max-depth error",
        f"{performance['physics_aware']['mean_relative_max_depth_error_percent']:.2f}%"
    )

with c4:
    st.metric(
        "MAE improvement",
        f"{performance['improvement']['MAE_percent']:.2f}%"
    )


# ============================================================
# LIVE PREDICTION
# ============================================================

st.markdown("---")
st.markdown("## 🚨 Live Flood Prediction")

st.info(
    "Enter a breach scenario within the ranges represented by the "
    "current training dataset. The FNO produces an approximate "
    "flood-depth field without rerunning the full simulation."
)

p1, p2 = st.columns(2)

with p1:

    length = st.number_input(
        "Reservoir / breach length (m)",
        min_value=float(normalization["length_min"]),
        max_value=float(normalization["length_max"]),
        value=1.50,
        step=0.05,
    )

    height = st.number_input(
        "Reservoir height (m)",
        min_value=float(normalization["height_min"]),
        max_value=float(normalization["height_max"]),
        value=1.75,
        step=0.05,
    )

with p2:

    peak_q = st.number_input(
        "Peak discharge Q (m³/s)",
        min_value=float(normalization["peak_Q_min"]),
        max_value=float(normalization["peak_Q_max"]),
        value=float(
            (
                normalization["peak_Q_min"]
                + normalization["peak_Q_max"]
            ) / 2
        ),
        step=0.01,
    )

    volume = st.number_input(
        "Hydrograph volume (m³)",
        min_value=float(normalization["volume_min"]),
        max_value=float(normalization["volume_max"]),
        value=float(
            (
                normalization["volume_min"]
                + normalization["volume_max"]
            ) / 2
        ),
        step=0.01,
    )


run_prediction = st.button(
    "🌊 RUN FNO FLOOD PREDICTION",
    type="primary",
    width="stretch",
)


if run_prediction:

    with st.spinner("Running physics-aware FNO..."):

        prediction, metadata = predict_flood(
            reservoir_length_m=length,
            reservoir_height_m=height,
            peak_q_m3s=peak_q,
            hydrograph_volume_m3=volume,
            output_name="live_prediction",
        )

    # --------------------------------------------------------
    # Physical checks
    # --------------------------------------------------------

    valid_prediction = prediction[mask]

    max_depth = float(valid_prediction.max())
    mean_depth = float(valid_prediction.mean())

    outside = prediction[~mask]

    outside_max = (
        float(np.abs(outside).max())
        if len(outside)
        else 0.0
    )

    negative_cells = int(
        np.sum(valid_prediction < 0)
    )

    # --------------------------------------------------------
    # Prototype trust score
    # --------------------------------------------------------

    score = 100.0

    if outside_max > 0.001:
        score -= 40

    if negative_cells > 0:
        score -= 20

    train_min = float(
        normalization["volume_min"]
    )

    if max_depth <= 0:
        score -= 40

    score = max(0.0, score)

    trust_status = (
        "TRUSTED"
        if score >= 80
        else "REVIEW"
        if score >= 60
        else "REJECT"
    )

    # --------------------------------------------------------
    # Impact screening
    # --------------------------------------------------------

    x_grid = np.load(
        DATA_DIR / "x_grid.npy"
    )

    y_grid = np.load(
        DATA_DIR / "y_grid.npy"
    )

    dx = abs(float(np.mean(np.diff(x_grid))))
    dy = abs(float(np.mean(np.diff(y_grid))))

    cell_area = dx * dy

    flooded = valid_prediction >= 0.01
    high = valid_prediction >= 0.20

    flooded_area = float(
        flooded.sum() * cell_area
    )

    high_area = float(
        high.sum() * cell_area
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.markdown("### Prediction Results")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Maximum depth",
            f"{max_depth:.4f} m"
        )

    with r2:
        st.metric(
            "Mean depth",
            f"{mean_depth:.4f} m"
        )

    with r3:
        st.metric(
            "Flooded area",
            f"{flooded_area:.1f} m²"
        )

    with r4:
        st.metric(
            "High-depth area",
            f"{high_area:.1f} m²"
        )

    st.markdown("### Trust Engine")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.metric(
            "Trust score",
            f"{score:.0f}/100"
        )

    with t2:
        st.metric(
            "Status",
            trust_status
        )

    with t3:
        st.metric(
            "Outside-domain depth",
            f"{outside_max:.6f} m"
        )

    # --------------------------------------------------------
    # Prediction map
    # --------------------------------------------------------

    st.markdown("### Predicted Flood Depth")

    fig, ax = plt.subplots(figsize=(12, 4))

    image = ax.imshow(
        prediction,
        origin="lower",
        aspect="auto"
    )

    ax.set_title(
        "Physics-aware FNO Flood-depth Prediction"
    )

    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    fig.colorbar(
        image,
        ax=ax,
        label="Flood depth (m)"
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # Save latest metadata in session
    # --------------------------------------------------------

    st.session_state["last_prediction"] = metadata

    st.success(
        "Prediction completed successfully."
    )


# ============================================================
# VALIDATION SCENARIOS
# ============================================================

st.markdown("---")
st.markdown("## Validation Scenarios")

scenario_rows = []

for s in validation_scenarios:

    scenario_rows.append({
        "Scenario": s["scenario"],
        "Trust": f"{s['trust']['score']:.0f}/100",
        "Status": s["trust"]["status"],
        "Maximum depth (m)":
            s["impact"]["maximum_depth_m"],
        "Mean depth (m)":
            s["impact"]["mean_depth_m"],
        "Flooded area (m²)":
            s["impact"][
                "flooded_area_gt_0p01m_m2"
            ],
        "High-severity area (m²)":
            s["impact"][
                "high_severity_area_ge_0p20m_m2"
            ],
    })

st.dataframe(
    pd.DataFrame(scenario_rows),
    width="stretch",
    hide_index=True,
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown("## Model Comparison")

comparison_df = pd.DataFrame([
    {
        "Model": "Baseline FNO",
        "MAE (m)": performance["baseline"]["MAE_m"],
        "RMSE (m)": performance["baseline"]["RMSE_m"],
        "Max-depth error (%)":
            performance["baseline"][
                "mean_relative_max_depth_error_percent"
            ],
        "Outside-mask (m)":
            performance["baseline"][
                "outside_mask_max_depth_m"
            ],
    },
    {
        "Model": "Physics-aware FNO",
        "MAE (m)": performance["physics_aware"]["MAE_m"],
        "RMSE (m)": performance["physics_aware"]["RMSE_m"],
        "Max-depth error (%)":
            performance["physics_aware"][
                "mean_relative_max_depth_error_percent"
            ],
        "Outside-mask (m)":
            performance["physics_aware"][
                "outside_mask_max_depth_m"
            ],
    },
])

st.dataframe(
    comparison_df,
    width="stretch",
    hide_index=True,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "JalRaksha is currently a controlled flume-domain prototype. "
    "FNO results are approximate surrogate predictions. "
    "Trust scores are engineering screening indicators, not "
    "certified probabilities. Real SAR validation requires a "
    "georeferenced flood event."
)
