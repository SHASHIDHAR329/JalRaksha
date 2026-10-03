from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Optional plotting backends are imported lazily.
PROJECT_ROOT = Path(r"C:\JalRaksha")
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = OUTPUT_DIR / "fno_dataset"
# Member-3 GIS prototype outputs.
# These are controlled/test layers, not verified Hirakud/Chiplima geometry.
GIS_DIR = PROJECT_ROOT / "s01_integration_work" / "member3_gis" / "gis" / "output"
GIS_FLOOD_EXTENT = GIS_DIR / "pipeline_flood_extent.tif"
GIS_HAZARD = GIS_DIR / "pipeline_hazard.tif"
GIS_IMPACT = GIS_DIR / "pipeline_impact.tif"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="JalRaksha | Flood Decision Support",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Visual system: white, graphite, one restrained blue accent.
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root{
        --ink:#172033;
        --muted:#667085;
        --line:#E4E7EC;
        --soft:#F7F8FA;
        --accent:#2457A6;
        --accent-soft:#EEF4FC;
        --good:#2F5D50;
        --warn:#8A6A1F;
    }
    .stApp { background:#FFFFFF; color:var(--ink); }
    .block-container { max-width:1220px; padding-top:1.15rem; padding-bottom:2rem; }
    [data-testid="stHeader"] { background:#FFFFFF; }
    h1,h2,h3,h4 { color:var(--ink)!important; letter-spacing:-0.02em; }
    .jr-subtitle { color:var(--muted); margin:.20rem 0 1rem 0; font-size:.94rem; }
    .jr-card {
        background:#FFFFFF;
        border:1px solid var(--line);
        border-radius:10px;
        padding:1rem 1.05rem;
        box-shadow:0 1px 2px rgba(16,24,40,.025);
    }
    .jr-section {
        font-size:.73rem;
        font-weight:750;
        text-transform:uppercase;
        letter-spacing:.08em;
        color:var(--muted);
        margin:.35rem 0 .45rem 0;
    }
    .jr-value { font-size:1.45rem; font-weight:740; color:var(--ink); }
    .jr-muted { color:var(--muted); font-size:.82rem; line-height:1.5; }
    .jr-callout {
        background:var(--accent-soft);
        border-left:3px solid var(--accent);
        padding:.72rem .85rem;
        border-radius:0 8px 8px 0;
        line-height:1.5;
    }
    .jr-status {
        display:inline-block;
        border:1px solid var(--line);
        background:var(--soft);
        padding:.28rem .52rem;
        border-radius:999px;
        font-size:.74rem;
        font-weight:720;
    }
    .jr-small { font-size:.76rem; color:var(--muted); }
    .jr-footer {
        border-top:1px solid var(--line);
        margin-top:2rem;
        padding-top:.85rem;
        color:var(--muted);
        font-size:.74rem;
        line-height:1.45;
    }
    div.stButton > button {
        border-radius:8px;
        border:1px solid var(--accent);
        background:var(--accent);
        color:#FFFFFF;
        font-weight:700;
    }
    div.stButton > button:hover {
        background:#1F4A8C;
        border-color:#1F4A8C;
    }
    [data-testid="stMetric"] {
        border:1px solid var(--line);
        border-radius:10px;
        padding:.72rem .8rem;
        background:#FFFFFF;
    }
    [data-testid="stMetricLabel"] { color:var(--muted)!important; }
    [data-testid="stMetricValue"] { color:var(--ink)!important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def display_optional_depth(value: Any) -> str:
    v = safe_float(value)
    return "Not available" if not math.isfinite(v) else f"{v:.4f} m"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def recursive_metric(obj: Any, names: tuple[str, ...], default: float) -> float:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower().replace("-", "_")
            if any(n in key for n in names):
                n = safe_float(v)
                if math.isfinite(n):
                    return n
            n = recursive_metric(v, names, float("nan"))
            if math.isfinite(n):
                return n
    elif isinstance(obj, list):
        for v in obj:
            n = recursive_metric(v, names, float("nan"))
            if math.isfinite(n):
                return n
    return default


# ---------------------------------------------------------------------
# Real project data
# ---------------------------------------------------------------------
MANIFEST = PROJECT_ROOT / "s01_integration_work" / "hirakud_dashboard_manifest.csv"
DASH_JSON = OUTPUT_DIR / "dashboard_data" / "jalraksha_dashboard.json"
Y_PATH = DATA_DIR / "Y.npy"
MASK_PATH = DATA_DIR / "mask.npy"
ROUTING_CSV = OUTPUT_DIR / "routing_stress_test" / "stress_test_results.csv"

manifest = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
if not manifest.empty and "scenario" not in manifest.columns:
    for candidate in ("scenario_id", "id", "Scenario"):
        if candidate in manifest.columns:
            manifest["scenario"] = manifest[candidate]
            break

if "scenario" in manifest.columns:
    manifest["scenario"] = manifest["scenario"].astype(str)

Y_DATA = None
FNO_MASK = None
if Y_PATH.exists():
    try:
        Y_DATA = np.load(Y_PATH)
        if MASK_PATH.exists():
            FNO_MASK = np.load(MASK_PATH).astype(bool)
    except Exception:
        Y_DATA = None

dashboard_data = load_json(DASH_JSON)

VERIFIED_MAE = recursive_metric(
    dashboard_data, ("physics_aware_mae",), 0.000366
)
VERIFIED_RMSE = recursive_metric(
    dashboard_data, ("physics_aware_rmse",), 0.001427
)
VERIFIED_MAX_ERR = recursive_metric(
    dashboard_data, ("physics_aware_max", "max_depth_error"), 5.56
)
VALIDATION_AGREEMENT = max(0.0, min(100.0, 100.0 - VERIFIED_MAX_ERR))

# Single deployment-facing number, shown only on Overview.
OPERATIONAL_CONFIDENCE = None
CONFIDENCE_LABEL = "Not established for Hirakud S01"


def scenario_number(s: str) -> int:
    digits = "".join(ch for ch in str(s) if ch.isdigit())
    return int(digits) if digits else 1


def scenario_row(selected: str) -> dict[str, Any]:
    if manifest.empty:
        return {}
    hit = manifest[manifest["scenario"].eq(selected)]
    return hit.iloc[0].to_dict() if not hit.empty else {}


def scenario_folder(num: int) -> Path:
    return OUTPUT_DIR / "dflowfm" / f"jalraksha_flume_scenario_{num:02d}"


def dflow_map_file(num: int) -> Path | None:
    # Canonical Hirakud S01 D-Flow FM result.
    if num == 1:
        hirakud = Path(r"C:\JalRaksha\s01_integration_work\hirakud_s01_map.nc")
        if hirakud.exists():
            return hirakud

    # Legacy flume scenarios remain available when their outputs exist.
    folder = scenario_folder(num)
    p = folder / "tidalflume_map.nc"
    return p if p.exists() else None


def first_matching_name(names: list[str], patterns: tuple[str, ...]) -> str | None:
    lowered = {n.lower(): n for n in names}
    for p in patterns:
        for low, original in lowered.items():
            if p in low:
                return original
    return None


# ---------------------------------------------------------------------
# Existing Member-1 prediction adapter. Backend is NOT modified.
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_predict_module():
    try:
        import predict_flood  # type: ignore
        return predict_flood
    except Exception:
        return None


def call_predict_flood(length_m: float, height_m: float, peak_q: float, volume_m3: float):
    module = load_predict_module()
    if module is None or not hasattr(module, "predict_flood"):
        raise RuntimeError("src/predict_flood.py could not be imported.")

    fn = module.predict_flood
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        low = name.lower()
        if "length" in low or low in {"l", "breach"}:
            kwargs[name] = length_m
        elif "height" in low or low == "h":
            kwargs[name] = height_m
        elif "peak" in low or "discharge" in low or low in {"q", "qpeak"}:
            kwargs[name] = peak_q
        elif "volume" in low:
            kwargs[name] = volume_m3

    required = [
        p for p in sig.parameters.values()
        if p.default is inspect._empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    missing = [p.name for p in required if p.name not in kwargs]
    if missing:
        raise RuntimeError(
            "Unsupported required inputs in predict_flood(): " + ", ".join(missing)
        )

    return fn(**kwargs)


def extract_prediction(result: Any) -> tuple[np.ndarray, dict[str, float]]:
    metrics: dict[str, float] = {}
    arr = None

    if isinstance(result, dict):
        for key in ("depth", "flood_depth", "prediction", "predicted_depth", "pred"):
            value = result.get(key)
            if isinstance(value, np.ndarray):
                arr = value
                break
        for key in ("max_depth_m", "max_depth"):
            if key in result:
                metrics["max_depth_m"] = safe_float(result[key])
        for key in ("mean_depth_m", "mean_depth"):
            if key in result:
                metrics["mean_depth_m"] = safe_float(result[key])
        for key in ("flooded_area_m2", "flooded_area"):
            if key in result:
                metrics["flooded_area_m2"] = safe_float(result[key])
    elif isinstance(result, np.ndarray):
        arr = result
    elif isinstance(result, (tuple, list)):
        for value in result:
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                arr = value
                break

    if arr is None:
        raise RuntimeError("No 2D flood-depth array was returned by predict_flood().")

    arr = np.asarray(arr, dtype=float).squeeze()
    if arr.ndim != 2:
        raise RuntimeError(f"Prediction has shape {arr.shape}; expected a 2D field.")

    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)

    metrics["max_depth_m"] = (
        metrics.get("max_depth_m", float("nan"))
        if math.isfinite(metrics.get("max_depth_m", float("nan")))
        else float(np.max(arr))
    )
    metrics["mean_depth_m"] = (
        metrics.get("mean_depth_m", float("nan"))
        if math.isfinite(metrics.get("mean_depth_m", float("nan")))
        else float(np.mean(arr))
    )

    if not math.isfinite(metrics.get("flooded_area_m2", float("nan"))):
        valid = np.isfinite(arr)
        if FNO_MASK is not None and FNO_MASK.shape == arr.shape:
            valid &= FNO_MASK
        # Area uses the current prototype grid convention.
        metrics["flooded_area_m2"] = float(np.sum(valid & (arr >= 0.01)))

    return arr, metrics


# ---------------------------------------------------------------------
# D-Flow FM viewer: uses actual scenario map output when its schema can
# be identified. No synthetic terrain is generated.
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_gis_raster(path_str: str):
    try:
        import rasterio

        with rasterio.open(path_str) as src:
            data = src.read(1)
            bounds = src.bounds
            crs = src.crs
            transform = src.transform

        return {
            "data": np.asarray(data),
            "bounds": bounds,
            "crs": str(crs),
            "transform": transform,
        }
    except Exception:
        return None
@st.cache_data(show_spinner=False)
def load_dflow_points(path_str: str):
    path = Path(path_str)

    # Fast path for the canonical Hirakud S01 dashboard view.
    snapshot = Path(r"C:\JalRaksha\s01_integration_work\hirakud_s01_dashboard_snapshot.npz")
    if path.name == "hirakud_s01_map.nc" and snapshot.exists():
        try:
            data = np.load(snapshot)
            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)
            depth = np.maximum(np.asarray(data["depth"], dtype=np.float32), 0.0)

            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(depth)
            x, y, depth = x[valid], y[valid], depth[valid]

            if x.size >= 20:
                meta = {
                    "x_name": "mesh2d_face_x",
                    "y_name": "mesh2d_face_y",
                    "z_name": "mesh2d_waterdepth",
                    "n": int(x.size),
                    "source": "hirakud_s01_dashboard_snapshot.npz",
                }
                return x, y, depth, meta
        except Exception:
            pass

    # Legacy/general NetCDF fallback.
    try:
        import xarray as xr  # type: ignore
    except Exception:
        return None

    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception:
        return None

    names = list(ds.variables)

    x_name = first_matching_name(
        names,
        ("mesh2d_face_x", "face_x", "mesh2d_node_x", "node_x", "x"),
    )
    y_name = first_matching_name(
        names,
        ("mesh2d_face_y", "face_y", "mesh2d_node_y", "node_y", "y"),
    )
    z_name = first_matching_name(
        names,
        (
            "mesh2d_waterdepth",
            "mesh2d_depth",
            "waterdepth",
            "water_depth",
            "mesh2d_s1",
            "s1",
        ),
    )

    if not x_name or not y_name or not z_name:
        ds.close()
        return None

    x = np.asarray(ds[x_name].values).squeeze()
    y = np.asarray(ds[y_name].values).squeeze()
    zraw = np.asarray(ds[z_name].values).squeeze()

    if zraw.ndim > 1:
        zraw = zraw[-1]

    x = x.ravel()
    y = y.ravel()
    z = np.asarray(zraw).ravel()

    n = min(x.size, y.size, z.size)
    x, y, z = x[:n], y[:n], z[:n]

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]

    if x.size < 20:
        ds.close()
        return None

    z = np.maximum(z, 0.0)

    meta = {
        "x_name": x_name,
        "y_name": y_name,
        "z_name": z_name,
        "n": int(x.size),
        "source": str(path),
    }
    ds.close()
    return x, y, z, meta

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = "scenario_01" if "scenario_01" in set(manifest["scenario"]) else (
        manifest["scenario"].iloc[0] if not manifest.empty else "scenario_02"
    )

if "prediction_by_scenario" not in st.session_state:
    st.session_state.prediction_by_scenario = {}


# ---------------------------------------------------------------------
# Header + navigation
# ---------------------------------------------------------------------
st.markdown("## JalRaksha")
st.markdown(
    '<div class="jr-subtitle">Physics-grounded flood decision support for rapid assessment and evacuation planning</div>',
    unsafe_allow_html=True,
)

pages = [
    "Overview",
    "Scenario Monitor",
    "Digital Twin",
    "Evacuation",
    "Validation",
]
page = st.radio("Section", pages, horizontal=True, label_visibility="collapsed", key="page")
st.divider()


# ---------------------------------------------------------------------
# Overview: the only place where operational confidence is shown.
# ---------------------------------------------------------------------
if page == "Overview":
    st.subheader("Current situation")

    selected = st.selectbox(
        "Simulation scenario",
        list(manifest["scenario"]) if not manifest.empty else ["scenario_02"],
        index=max(0, (list(manifest["scenario"]).index(st.session_state.selected_scenario)
                     if st.session_state.selected_scenario in set(manifest["scenario"]) else 0)),
        key="selected_scenario_widget",
    )
    st.session_state.selected_scenario = selected
    row = scenario_row(selected)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="jr-card"><div class="jr-section">S01 physics status</div>'
            '<div class="jr-value">24h COMPLETE</div>'
            '<div class="jr-small">D-Flow FM hydraulic run</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="jr-card"><div class="jr-section">Hydraulic QA</div>'
            '<div class="jr-value">CONDITIONAL PASS</div>'
            '<div class="jr-small">Conservation and numerical checks completed</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="jr-card"><div class="jr-section">AI trust state</div>'
            '<div class="jr-value">LAB ONLY</div>'
            '<div class="jr-small">FNO blocked for Hirakud-scale inputs</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Selected scenario")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Modeled path length", f"{safe_float(row.get('L', row.get('length', 0))) / 1000:,.2f} km")
    with s2:
        st.metric("Initial reservoir level", f"{safe_float(row.get('H', row.get('height', 0))):,.3f} m")
    with s3:
        st.metric("Peak discharge", f"{safe_float(row.get('PeakQ', row.get('peak_q', 0))):,.0f} m3/s")
    with s4:
        st.metric("Hydrograph volume", f"{safe_float(row.get('Volume', row.get('volume', 0))) / 1e9:.2f} billion m3")

    st.markdown(
        '<div class="jr-callout"><b>How to read this page:</b> the scenario is selected from the actual '
        'simulation set. The dashboard reports the hydraulic inputs first, then shows the model output and '
        'decision-support layers. Weather is external context and is not an FNO input.</div>',
        unsafe_allow_html=True,
    )

    # Weather: one compact context block.
    with st.expander("Current weather context", expanded=False):
        w1, w2 = st.columns(2)
        with w1:
            lat = st.number_input("Latitude", value=12.9716, format="%.4f")
        with w2:
            lon = st.number_input("Longitude", value=77.5946, format="%.4f")
        if st.button("Refresh weather"):
            try:
                from weather import get_current_weather
                weather = get_current_weather(float(lat), float(lon))
                current = weather.get("current", {})
                st.write(
                    f"Temperature: {safe_float(current.get('temperature_2m')):.1f} Â°C  |  "
                    f"Humidity: {safe_float(current.get('relative_humidity_2m')):.0f}%  |  "
                    f"Precipitation: {safe_float(current.get('precipitation')):.1f} mm  |  "
                    f"Wind: {safe_float(current.get('wind_speed_10m')):.1f} km/h"
                )
                st.caption(
                    f"Source response time: {weather.get('current', {}).get('time', 'unknown')} "
                    f"({weather.get('timezone', 'local')}). Context only."
                )
            except Exception as exc:
                st.warning(f"Weather context unavailable: {exc}")

    st.markdown("### System evidence")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown(
            f'<div class="jr-card"><div class="jr-section">FNO laboratory validation</div>'
            f'<b>FNO MAE (lab)</b> {VERIFIED_MAE:.6f} m<br>'
            f'<b>FNO RMSE (lab)</b> {VERIFIED_RMSE:.6f} m<br>'
            f'<b>Max-depth error (lab)</b> {VERIFIED_MAX_ERR:.2f}%</div>',
            unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            '<div class="jr-card"><div class="jr-section">What is not claimed</div>'
            'No certified operational safety, no real-road evacuation validation, and no satellite-event '
            'accuracy claim until georeferenced event data are available.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# Scenario Monitor
# ---------------------------------------------------------------------
elif page == "Scenario Monitor":
    st.subheader("Scenario monitor")
    st.caption("Hirakud S01 hydraulic case. Inputs come from the verified 24-hour D-Flow FM integration run.")

    options = list(manifest["scenario"]) if not manifest.empty else ["scenario_02"]
    selected = st.selectbox("Scenario", options, key="selected_scenario_widget")
    row = scenario_row(selected)
    st.session_state.selected_scenario = selected

    length_m = safe_float(row.get("reservoir_length_m", row.get("L", row.get("length", 1.5))), 1.5)
    height_m = safe_float(row.get("reservoir_height_m", row.get("H", row.get("height", 1.75))), 1.75)
    peak_q = safe_float(row.get("peak_Q_m3s", row.get("PeakQ", row.get("peak_q", 0.62))), 0.62)
    volume_m3 = safe_float(row.get("hydrograph_volume_m3", row.get("Volume", row.get("volume", 0.65))), 0.65)
    scenario_num = scenario_number(selected)

    a, b = st.columns(2)
    with a:
        st.markdown("#### Hydraulic input")
        st.dataframe(
            pd.DataFrame(
                {
                    "Parameter": ["Length", "Height", "Peak discharge", "Hydrograph volume"],
                    "Value": [
                        f"{length_m:.2f} m",
                        f"{height_m:.2f} m",
                        f"{peak_q:.4f} mÂ³/s",
                        f"{volume_m3:.4f} mÂ³",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with b:
        st.markdown("#### Reference simulation summary")
        st.dataframe(
            pd.DataFrame(
                {
                    "Measure": ["Reference max depth", "Reference mean depth"],
                    "Value": [
                        display_optional_depth(
                            row.get("MaxDepth", row.get("max_depth", float("nan")))
                        ),
                        display_optional_depth(
                            row.get("MeanDepth", row.get("mean_depth", float("nan")))
                        ),
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    # --------------------------------------------------------------
    # Domain-aware FNO trust gate
    # --------------------------------------------------------------
    fno_ranges = {
        "Length": (1.0, 2.0, length_m, "m"),
        "Height": (1.5, 2.0, height_m, "m"),
        "Peak discharge": (0.54074, 0.87118, peak_q, "m3/s"),
        "Hydrograph volume": (0.5167810983888199, 0.8387989751090003, volume_m3, "m3"),
    }

    fno_out_of_domain = [
        f"{name}: {value:g} {unit} not in [{vmin:g}, {vmax:g}]"
        for name, (vmin, vmax, value, unit) in fno_ranges.items()
        if value < vmin or value > vmax
    ]

    if fno_out_of_domain:
        st.warning(
            "AI model status: OUTSIDE TRAINED DOMAIN. "
            "The current FNO is laboratory/flume-domain only. "
            "For this scenario, use the D-Flow FM hydraulic result."
        )
        with st.expander("Why the AI result is blocked", expanded=False):
            st.write("\n".join(fno_out_of_domain))
        run_fno = False
    else:
        st.success(
            "AI model status: IN TRAINED DOMAIN. "
            "FNO inference is permitted for this laboratory/flume scenario."
        )
        run_fno = st.button("Run physics-aware FNO for this scenario")

    if run_fno:
        try:
            with st.spinner("Computing the flood-depth field..."):
                raw = call_predict_flood(length_m, height_m, peak_q, volume_m3)
                depth, metrics = extract_prediction(raw)
            st.session_state.prediction_by_scenario[selected] = (depth, metrics)
            st.success("Prediction completed.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if selected in st.session_state.prediction_by_scenario:
        depth, metrics = st.session_state.prediction_by_scenario[selected]

        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Predicted maximum depth", f"{metrics['max_depth_m']:.4f} m")
        with p2:
            st.metric("Predicted mean depth", f"{metrics['mean_depth_m']:.4f} m")
        with p3:
            st.metric("Predicted flooded cells", f"{int(np.sum(depth >= 0.01)):,}")

        try:
            import plotly.graph_objects as go
            fig = go.Figure(
                go.Heatmap(
                    z=depth,
                    colorbar=dict(title="Depth (m)"),
                    hovertemplate="x=%{x}<br>y=%{y}<br>depth=%{z:.4f} m<extra></extra>",
                )
            )
            fig.update_layout(
                height=420,
                margin=dict(l=5, r=5, t=10, b=10),
                xaxis_title="Computational X",
                yaxis_title="Computational Y",
            )
            st.plotly_chart(fig, width="stretch")
        except Exception as exc:
            st.error(f"Flood field display failed: {exc}")

        st.markdown(
            f'<div class="jr-card"><div class="jr-section">Decision interpretation</div>'
            f'For <b>{selected}</b>, the model predicts a maximum depth of '
            f'<b>{metrics["max_depth_m"]:.3f} m</b>. Use the impact and evacuation views '
            f'together with verification evidence before treating the result as an operational decision.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Run the scenario to display the current model output.")
    #---------------------------------------------------------------------
    # Member-3 GIS prototype layer
    # ---------------------------------------------------------------------
    st.subheader("GIS prototype layer")
    st.caption(
        "Controlled/test GIS dataset from Member 3. "
        "Not verified Hirakud/Chiplima event geometry."
    )

    gis_layers = {
        "Flood extent": GIS_FLOOD_EXTENT,
        "Hazard": GIS_HAZARD,
        "Impact": GIS_IMPACT,
    }

    gis_choice = st.selectbox(
        "GIS layer",
        list(gis_layers.keys()),
        key="gis_prototype_layer",
    )

    gis_path = gis_layers[gis_choice]

    if not gis_path.exists():
        st.warning(f"GIS layer file not found: {gis_path}")
    else:
        gis_data = load_gis_raster(str(gis_path))

        if gis_data is None:
            st.error("The GIS raster could not be read.")
        else:
            raster = gis_data["data"]
            bounds = gis_data["bounds"]

            g1, g2, g3 = st.columns(3)

            with g1:
                st.metric("Raster size", f"{raster.shape[1]} × {raster.shape[0]}")

            with g2:
                st.metric("CRS", gis_data["crs"])

            with g3:
                st.metric("Maximum value", f"{float(np.max(raster)):.2f}")

            try:
                import plotly.graph_objects as go

                fig = go.Figure(
                    go.Heatmap(
                        z=raster,
                        colorbar=dict(title=gis_choice),
                        hovertemplate=(
                            "column=%{x}<br>"
                            "row=%{y}<br>"
                            "value=%{z}<extra></extra>"
                        ),
                    )
                )

                fig.update_layout(
                    height=480,
                    margin=dict(l=5, r=5, t=10, b=10),
                    xaxis_title="Raster column",
                    yaxis_title="Raster row",
                )

                st.plotly_chart(fig, width="stretch")

            except Exception as exc:
                st.error(f"GIS layer display failed: {exc}")

            st.markdown(
                f'<div class="jr-card">'
                f'<div class="jr-section">GIS evidence status</div>'
                f'<b>Layer:</b> {gis_choice}<br>'
                f'<b>CRS:</b> {gis_data["crs"]}<br>'
                f'<b>Bounds:</b> '
                f'{bounds.left:.4f}, {bounds.bottom:.4f} → '
                f'{bounds.right:.4f}, {bounds.top:.4f}<br><br>'
                f'This layer demonstrates the Member-3 GIS processing workflow '
                f'using controlled/test data. It is not presented as verified '
                f'Hirakud or Chiplima georeferenced flood data.'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------
# Digital Twin
# ---------------------------------------------------------------------
elif page == "Digital Twin":
    st.subheader("Hydraulic digital twin")
    st.caption("Uses the actual D-Flow FM scenario map output when the file schema is available.")

    options = list(manifest["scenario"]) if not manifest.empty else ["scenario_02"]
    selected = st.selectbox("Scenario", options, key="selected_scenario_widget")
    num = scenario_number(selected)
    nc_path = dflow_map_file(num)

    if nc_path is None:
        st.info(f"No D-Flow FM map file was found for {selected}.")
    else:
        data = load_dflow_points(str(nc_path))
        if data is None:
            st.warning("The D-Flow file exists, but its variables could not be mapped automatically.")
        else:
            x, y, z, meta = data

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Displayed cells/points", f"{meta['n']:,}")
            with m2:
                st.metric("Maximum displayed depth", f"{np.max(z):.4f} m")
            with m3:
                source_label = (
                    "S01 dashboard snapshot"
                    if meta.get("source") == "hirakud_s01_dashboard_snapshot.npz"
                    else "D-Flow FM"
                )
                st.metric("Data source", source_label)

            try:
                import plotly.graph_objects as go
                from scipy.spatial import Delaunay

                pts = np.column_stack([x, y])
                tri = Delaunay(pts)

                fig = go.Figure(
                    data=[
                        go.Mesh3d(
                            x=x,
                            y=y,
                            z=z,
                            i=tri.simplices[:, 0],
                            j=tri.simplices[:, 1],
                            k=tri.simplices[:, 2],
                            intensity=z,
                            colorscale="Blues",
                            opacity=0.92,
                            flatshading=False,
                            hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<br>Depth=%{z:.3f} m<extra></extra>",
                            colorbar=dict(title="Depth (m)"),
                        )
                    ]
                )
                
                
                fig.update_layout(
    height=600,
    margin=dict(l=0, r=0, t=8, b=0),
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Water depth (m)",
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=1.55, z=1.05)),
    ),
)

                st.plotly_chart(fig, width="stretch")
            except Exception as exc:
                st.error(f"Digital-twin rendering failed: {exc}")

            st.markdown(
                f'<div class="jr-card"><div class="jr-section">Source record</div>'
                f'<b>Scenario:</b> {selected}<br>'
                f'<b>Original hydraulic source:</b> {nc_path.name}<br>'
                f'<b>Dashboard rendering asset:</b> {meta.get("source", "D-Flow FM")}<br>'
                f'<b>X variable:</b> {meta["x_name"]}<br>'
                f'<b>Y variable:</b> {meta["y_name"]}<br>'
                f'<b>Hydraulic variable:</b> {meta["z_name"]}<br>'
                f'<span class="jr-muted">The view is a final hydraulic-state visualization; it is not a fabricated transient animation.</span></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------
# Evacuation
# ---------------------------------------------------------------------
elif page == "Evacuation":
    st.subheader("Trust-aware evacuation routing")
    st.caption("Current Member-1 routing proof is a controlled algorithm stress test; real GIS roads are added later.")
    st.markdown(
    '<div class="jr-card">'
    '<div class="jr-section">Trust-aware routing logic</div>'
    '<b>1. Assess trust</b> → Evaluate the reliability of available corridor information.<br>'
    '<b>2. Adjust routing cost</b> → Lower-trust corridors receive a higher adaptive cost.<br>'
    '<b>3. Select corridor</b> → The route can shift from the shortest corridor to a longer, more trusted corridor.<br><br>'
    '<b>Prototype status:</b> Controlled algorithm demonstration; real georeferenced road and shelter data are not yet integrated.'
    '</div>',
    unsafe_allow_html=True,
)
    if ROUTING_CSV.exists():
        route_df = pd.read_csv(ROUTING_CSV)
        if not route_df.empty:
            cols = [
                c for c in [
                    "trust_score",
                    "selected_corridor",
                    "physical_distance_m",
                    "adaptive_routing_cost",
                ] if c in route_df.columns
            ]
            display = route_df[cols].copy()
            rename = {
                "trust_score": "Trust level",
                "selected_corridor": "Selected corridor",
                "physical_distance_m": "Distance (m)",
                "adaptive_routing_cost": "Adaptive cost",
            }
            display = display.rename(columns=rename)
            st.dataframe(display, hide_index=True, width="stretch")

            st.markdown(
                '<div class="jr-card"><div class="jr-section">Observed routing behavior</div>'
                'As confidence falls, the routing cost of uncertain corridors increases, so the prototype '
                'can move to a longer but more trusted corridor. The uncertainty values in this current test '
                'are synthetic algorithm parameters, not measured road uncertainty.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Routing stress-test results are empty.")
    else:
        st.info("Run src/trust_route_stress_test.py to populate the routing demonstration.")

    if ROUTING_PNG := OUTPUT_DIR / "routing_stress_test" / "stress_test_routes.png":
        if ROUTING_PNG.exists():
            st.image(str(ROUTING_PNG), width="stretch")


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
elif page == "Validation":
    st.subheader("FNO laboratory validation")
    st.caption("Verified Member-2 physics-aware FNO metrics for the controlled laboratory/flume domain. These are not Hirakud validation metrics.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("FNO MAE (lab)", f"{VERIFIED_MAE:.6f} m")
    with c2:
        st.metric("FNO RMSE (lab)", f"{VERIFIED_RMSE:.6f} m")
    with c3:
        st.metric("Max-depth error (lab)", f"{VERIFIED_MAX_ERR:.2f}%")

    st.markdown("### Held-out laboratory prediction check")

    validation_rows = []
    eval_dir = OUTPUT_DIR / "fno_model_physics" / "evaluation"
    for sid in ("scenario_02", "scenario_09"):
        pred_path = eval_dir / f"{sid}_prediction.npy"
        truth_path = eval_dir / f"{sid}_truth.npy"
        if pred_path.exists() and truth_path.exists():
            try:
                pred = np.asarray(np.load(pred_path), dtype=float).squeeze()
                truth = np.asarray(np.load(truth_path), dtype=float).squeeze()
                mae = float(np.mean(np.abs(pred - truth)))
                rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
                max_err = float(abs(np.max(pred) - np.max(truth)))
                validation_rows.append(
                    {
                        "Scenario": sid,
                        "Average Prediction Error (m)": mae,
                        "Prediction Error (m)": rmse,
                        "Predicted max (m)": float(np.max(pred)),
                        "Reference max (m)": float(np.max(truth)),
                        "Max error (m)": max_err,
                    }
                )
            except Exception:
                pass

    if validation_rows:
        st.dataframe(
            pd.DataFrame(validation_rows),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Held-out prediction files were not available.")

    st.markdown(
        '<div class="jr-callout"><b>Interpretation:</b> the current evidence supports a controlled prototype. '
        'Real-world deployment would require georeferenced flood observations, road data, and event-level validation.</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="jr-footer">JalRaksha Â· Member-1 dashboard layer Â· Existing hydraulic/FNO/routing modules preserved. '
    'Weather is contextual. Current routing demonstration is algorithmic until real georeferenced road data are integrated.</div>',
    unsafe_allow_html=True,
)


st.markdown("""
<style>
.jr-title {
    display: block !important;
    width: auto !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow: visible !important;

    margin: 0 0 12px 0 !important;
    padding: 0 !important;

    color: #172033 !important;
    background: transparent !important;

    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 38px !important;
    font-weight: 800 !important;
    line-height: 1.25 !important;
    letter-spacing: 0.04em !important;

    white-space: nowrap !important;
    text-align: left !important;

    transform: none !important;
    clip: auto !important;
    clip-path: none !important;
}
</style>
""", unsafe_allow_html=True)





