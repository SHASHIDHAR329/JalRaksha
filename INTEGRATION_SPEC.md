# JalRaksha Integration Specification

## Project
JalRaksha - Physics-Aware AI Flood Prediction Prototype

## Current Member 1 Pipeline

DualSPHysics
    ->
FlowTool
    ->
Clean Hydrograph Q(t)
    ->
D-Flow FM
    ->
Flood-depth dataset
    ->
Regular-grid conversion
    ->
Physics-aware FNO
    ->
Trust Engine
    ->
Impact Analysis
    ->
Streamlit Dashboard


# ============================================================
# MEMBER 1 -> MEMBER 2
# ============================================================

## Model

Physics-aware FNO checkpoint:

C:\JalRaksha\outputs\fno_model_physics\fno_physics_best.pt


## Dataset

C:\JalRaksha\outputs\fno_dataset


## Input array

X.npy

Shape:
(10, 7, 16, 128)


## Output array

Y.npy

Shape:
(10, 1, 16, 128)


## Mask

mask.npy

Shape:
(16, 128)


## Normalization

normalization.json


## Inference interface

from predict_flood import predict_flood

prediction, metadata = predict_flood(
    reservoir_length_m,
    reservoir_height_m,
    peak_q_m3s,
    hydrograph_volume_m3
)


## Returned prediction

prediction:
2D flood-depth grid, shape (16, 128), units metres


## Returned metadata

metadata contains:

reservoir_length_m
reservoir_height_m
peak_q_m3s
hydrograph_volume_m3
grid_shape
maximum_depth_m
mean_depth_m
device
model
prediction_file


# ============================================================
# MEMBER 2 RESPONSIBILITIES
# ============================================================

Member 2 works on:

- FNO architecture experiments
- Training improvements
- Hyperparameter experiments
- Additional AI validation
- Model explanation
- Optional uncertainty estimation

Member 2 must preserve the inference interface unless
a new interface is explicitly agreed.


# ============================================================
# MEMBER 1 -> MEMBER 3
# ============================================================

Member 3 receives:

- x_grid
- y_grid
- flood_depth_grid
- maximum_depth_m
- mean_depth_m
- trust_score
- trust_status
- flooded_area_m2
- high_severity_area_m2


## Grid files

C:\JalRaksha\outputs\fno_dataset\x_grid.npy

C:\JalRaksha\outputs\fno_dataset\y_grid.npy


# ============================================================
# TRUST ENGINE
# ============================================================

Current prototype trust outputs:

trust_score
trust_status
outside_mask_max_depth_m
proxy_ratio


Current validated held-out cases:

scenario_02 -> 100/100 TRUSTED
scenario_09 -> 100/100 TRUSTED


Important:

Trust score is a prototype engineering screening layer.
It is NOT a certified probability of prediction correctness.


# ============================================================
# IMPACT ANALYSIS
# ============================================================

Current outputs:

C:\JalRaksha\outputs\impact_analysis\flood_impact_summary.csv

C:\JalRaksha\outputs\impact_analysis\flood_impact_summary.json


Prototype depth screening classes:

Dry       < 0.01 m
Low       0.01 - 0.10 m
Moderate  0.10 - 0.20 m
High      0.20 - 0.30 m
Severe    >= 0.30 m


These are prototype screening classes and are not official
regulatory flood-hazard thresholds.


# ============================================================
# DASHBOARD
# ============================================================

Application:

C:\JalRaksha\src\app.py


Run:

python -m streamlit run C:\JalRaksha\src\app.py


Local address:

http://localhost:8501


Dashboard data:

C:\JalRaksha\outputs\dashboard_data\jalraksha_dashboard.json


# ============================================================
# SAR VALIDATION
# ============================================================

SAR validator:

C:\JalRaksha\src\validate_sar.py


Input directory:

C:\JalRaksha\inputs\sar


Expected future files:

fno_flood_mask.tif
sentinel1_flood_mask.tif


Important:

The current flume coordinates are laboratory coordinates.
They must NOT be presented as a real geographic Sentinel-1
location.

Real SAR validation requires a real georeferenced flood event.


# ============================================================
# CURRENT MODEL RESULTS
# ============================================================

Physics-aware FNO validation:

MAE:
0.000366 m

RMSE:
0.001427 m

Mean relative maximum-depth error:
5.56 %

Compared with baseline FNO:

MAE improvement:
9.73 %

RMSE improvement:
4.36 %

Maximum-depth-error improvement:
4.69 %

Physics-aware FNO:
Outside-mask depth = 0
Negative prediction cells = 0


# ============================================================
# CURRENT DATASET
# ============================================================

10 scenarios exist.

Manifest:

C:\JalRaksha\outputs\dataset_manifest.csv


FNO-ready dataset:

C:\JalRaksha\outputs\fno_dataset


# ============================================================
# IMPORTANT DEVELOPMENT RULE
# ============================================================

Do NOT regenerate the SPH simulations unless a new dataset
experiment is intentionally planned.

Do NOT replace the working physics-aware checkpoint without
saving the previous model.

Do NOT fabricate geographic coordinates for SAR validation.

Do NOT call the current prototype results operational flood
forecasting accuracy.


# ============================================================
# NEXT INTEGRATION TASKS
# ============================================================

1. Member 2 -> finalize AI/inference interface.
2. Member 3 -> connect GIS/map visualization.
3. Member 1 -> connect Trust Engine to live inference.
4. Member 1 + Member 3 -> integrate impact outputs.
5. Add real georeferenced SAR validation.
6. Final end-to-end demo.
7. Final testing and backup.

# ============================================================
# PROJECT CHECKPOINT
# ============================================================

Member 1 foundation:
approximately 90%+ complete.

Current immediate work:
integration and final prototype polishing.
