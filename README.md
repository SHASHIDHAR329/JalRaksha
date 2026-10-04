# JalRaksha 🌊

### Physics-Grounded, Trust-Aware Flood Decision Support for Dam-Break Inundation

> **Smart India Hackathon 2026 — Problem Statement 161**
> **Dam Break Inundation Modelling Using Hydrodynamic Modelling of any River**

JalRaksha is a prototype flood decision-support platform that combines **physics-based hydrodynamic modelling, AI-based flood surrogates, trust assessment, GIS analysis, adaptive evacuation routing, and interactive visualization** into a unified workflow.

The system is built around a simple principle:

> **A flood prediction should be accompanied by evidence about its physical basis, model applicability, and uncertainty.**

---

# 🎯 Problem

Dam-break and natural-dam/lake failure events can generate rapidly propagating floods with severe downstream consequences.

Effective flood decision support requires more than a single predicted water-depth value. Emergency responders need to understand:

* How flooding propagates downstream
* Water depth and flow velocity
* Spatial flood extent and hazard
* Potential impact areas
* Whether an AI surrogate is appropriate for a given scenario
* How uncertainty can influence evacuation decisions

JalRaksha connects these stages into a single decision-support workflow.

---

# 🧠 System Architecture

```text
                    FLOOD SCENARIO
                         │
                         ▼
              ┌─────────────────────┐
              │ Flood Conditions    │
              │ Reservoir / Boundary│
              │ Scenario Parameters │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Physics Simulation  │
              │     D-Flow FM       │
              │ Hydrodynamic Model  │
              └──────────┬──────────┘
                         │
                 Depth / Velocity
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   ┌─────────────────┐      ┌─────────────────┐
   │ AI Flood        │      │ GIS Processing  │
   │ Surrogate (FNO) │      │                 │
   └────────┬────────┘      └────────┬────────┘
            │                        │
            ▼                        ▼
   ┌─────────────────┐      ┌─────────────────┐
   │ AI Trust &      │      │ Flood Extent    │
   │ Applicability   │      │ Hazard / Impact │
   └────────┬────────┘      └────────┬────────┘
            │                        │
            └────────────┬───────────┘
                         ▼
              ┌─────────────────────┐
              │ Decision Support    │
              │ Risk + Evacuation   │
              │ Trust-Aware Routing │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ JalRaksha Dashboard │
              │ Scenario Monitor    │
              │ Digital Twin        │
              │ GIS / Validation   │
              │ Evacuation         │
              └─────────────────────┘
```

---

# 🔬 Core Components

## 1. Physics-Based Hydrodynamic Modelling

JalRaksha uses **D-Flow FM (Flexible Mesh)** from Deltares as its physics-based hydraulic modelling component.

The model provides numerical information including:

* Water depth
* Water level
* Flow velocity
* Flood propagation
* Downstream hydraulic response

The current S01 demonstration contains a **24-hour D-Flow FM simulation** over a downstream domain of approximately **28.29 km**.

The hydraulic workflow has undergone numerical and conservation checks and is documented as a **conditional QA pass** based on the current modelling configuration.

### Official Technical Reference

**Deltares — D-Flow FM User Manual**

https://content.oss.deltares.nl/delft3d/D-Flow_FM_User_Manual.pdf

---

# 🤖 2. AI Flood Surrogate

JalRaksha incorporates a **Fourier Neural Operator (FNO)** as an AI-based flood surrogate.

The objective is to investigate whether learned surrogate models can provide rapid flood-field estimation while retaining a connection to physics-based simulation.

The current FNO evidence is based on a **laboratory/flume-domain validation dataset**.

The dashboard therefore includes an explicit AI applicability state:

> **AI Trust State: LAB ONLY**

This prevents the surrogate from being presented outside its demonstrated modelling domain.

---

# 🛡️ 3. Trust-Aware AI

A key innovation of JalRaksha is explicit **AI applicability and trust assessment**.

Instead of treating every AI prediction as equally reliable, the system considers:

* Model applicability
* Validation evidence
* Prediction behaviour
* Uncertainty
* Physics-based reference information

The current laboratory FNO evidence includes:

* **MAE:** ~0.000366 m
* **RMSE:** ~0.001427 m
* **Maximum depth error:** ~5.84%

These metrics represent the laboratory validation domain and are used to demonstrate the trust-assessment concept.

---

# 🗺️ 4. GIS Flood Interpretation

The GIS module converts flood-depth information into spatial decision-support layers.

```text
Flood Depth Raster
        │
        ├──► Flood Extent
        │
        ├──► Hazard Map
        │
        └──► Impact Map
```

### Flood Extent

Identifies flooded and non-flooded areas.

### Hazard Map

The prototype classifies flood depth into:

```text
0              → No flood
0 < depth ≤ 2  → Low
2 < depth ≤ 3  → Medium
depth > 3      → High
```

### Impact Map

Provides simplified impact classes based on flood depth.

The current GIS workflow demonstrates the transformation of hydraulic information into spatial decision-support layers and provides the foundation for integration with richer geospatial datasets.

---

# 🚨 5. Trust-Aware Evacuation Routing

JalRaksha demonstrates an evacuation-routing approach where **corridor reliability can influence route selection**.

Traditional routing may prioritize:

> **Shortest route**

JalRaksha demonstrates:

> **Adaptive routing considering corridor trust**

```text
Corridor Information
        │
        ▼
    Trust Score
        │
        ▼
 Adaptive Route Cost
        │
        ├──► Shorter corridor
        │
        └──► More trusted corridor
```

As corridor confidence decreases, its adaptive routing cost increases.

The prototype can therefore shift from a shorter corridor to a **longer but more trusted corridor** when reliability becomes more important.

This establishes the foundation for future integration with real road, shelter, blockage and accessibility data.

---

# 🌐 6. Contextual Weather

JalRaksha includes weather information as **external contextual information** for situational awareness.

Weather is kept conceptually separate from the current FNO input pipeline, allowing the dashboard to distinguish between:

* Hydraulic evidence
* AI prediction
* External context
* Trust assessment

---

# 🧊 7. 3D Hydraulic Digital Twin

The dashboard provides an interactive **3D hydraulic Digital Twin** for visual interpretation of the simulated flood field.

It visualizes:

* Hydraulic locations
* Water depth
* Water-level information
* Spatial variation of hydraulic results

A lightweight dashboard snapshot is used for interactive rendering while the original hydraulic outputs remain the underlying simulation evidence.

The current dashboard snapshot contains approximately **4,913 displayed hydraulic points/cells**.

---

# 📊 Integrated Dashboard

JalRaksha provides a Streamlit-based interface containing:

### Overview

* Hydraulic status
* Hydraulic QA
* AI trust state
* Scenario information
* Weather context
* Validation evidence

### Scenario Monitor

* Hydraulic scenario information
* Flood outputs
* GIS layers
* Flood extent
* Hazard
* Impact

### Digital Twin

Interactive 3D hydraulic visualization.

### Validation

Displays available hydraulic and AI validation evidence together with their applicable domains.

### Evacuation

Demonstrates trust-aware adaptive routing.

---

# 🔬 Evidence-Driven Design

JalRaksha separates different sources of information rather than presenting them as a single undifferentiated prediction.

| Layer            | Role                                          |
| ---------------- | --------------------------------------------- |
| **Physics**      | Numerical hydraulic evidence from D-Flow FM   |
| **AI**           | Rapid surrogate modelling research            |
| **Trust**        | Model applicability and uncertainty awareness |
| **GIS**          | Spatial interpretation of flood information   |
| **Routing**      | Decision-support for evacuation               |
| **Digital Twin** | Interactive visualization                     |
| **Weather**      | External situational context                  |

This architecture allows future improvements to individual components without replacing the complete system.

---

# 🚀 Current Status & Future Improvements

JalRaksha currently demonstrates an **end-to-end integrated prototype** connecting hydrodynamics, AI, trust assessment, GIS, evacuation routing and visualization.

The next development stage focuses on strengthening the system through:

* **Higher-fidelity geospatial integration** using verified event and infrastructure datasets
* **Expanded AI validation** across larger and more diverse hydraulic scenarios
* **Operational decision-support integration** with real road, shelter, population and accessibility information

These improvements will progressively move JalRaksha toward a more **validated, scalable and deployment-oriented flood decision-support platform**.

---

# 📁 Project Structure

```text
JalRaksha/
│
├── src/
│   ├── app.py
│   ├── predict_flood.py
│   ├── flood3d.py
│   ├── evacuation_router.py
│   ├── robust_flood_envelope.py
│   ├── robust_flood_impact.py
│   ├── trust_route_demo.py
│   ├── trust_route_stress_test.py
│   └── weather.py
│
├── s01_integration_work/
│   ├── member3_gis/
│   │   ├── gis/
│   │   ├── impact/
│   │   ├── trust/
│   │   └── validation/
│   │
│   ├── reports/
│   │   └── checkpoints/
│   │
│   ├── hirakud_s01.mdu
│   ├── hirakud_s01.bc
│   ├── hirakud_s01.ext
│   ├── hirakud_upstream.pli
│   └── chiplima_downstream.pli
│
├── .streamlit/
│   └── config.toml
│
├── .gitignore
└── README.md
```

---

# ⚙️ Technology Stack

| Layer             | Technology              |
| ----------------- | ----------------------- |
| Dashboard         | Streamlit               |
| Programming       | Python                  |
| Hydrodynamics     | Deltares D-Flow FM      |
| AI Surrogate      | Fourier Neural Operator |
| Numerical Data    | NetCDF / NumPy          |
| GIS Processing    | Rasterio / GeoTIFF      |
| Data Processing   | Pandas / NumPy          |
| Visualization     | Plotly / Streamlit      |
| Routing Prototype | Python                  |
| Version Control   | Git / GitHub            |

---

# 🚀 Getting Started

## Clone the repository

```bash
git clone https://github.com/SHASHIDHAR329/JalRaksha.git
cd JalRaksha
```

## Create the environment

```bash
conda create -n jalraksha python=3.11
conda activate jalraksha
```

## Install dashboard dependencies

```bash
pip install streamlit numpy pandas plotly rasterio
```

Additional dependencies may be required for individual AI, GIS and modelling workflows.

## Run the dashboard

```bash
streamlit run src/app.py
```

The dashboard will normally be available at:

```text
http://localhost:8501
```

---

# 🧪 GIS Prototype

The controlled GIS workflow can be reproduced with:

```bash
cd s01_integration_work/member3_gis

python gis/scripts/create_test_flood_map.py
python gis/scripts/run_gis_pipeline.py
```

The pipeline generates:

```text
pipeline_flood_extent.tif
pipeline_hazard.tif
pipeline_impact.tif
```

These layers demonstrate the GIS processing and visualization workflow and provide the basis for integration with future verified geospatial datasets.

---

# 📚 Technical References

## 1. D-Flow FM Hydrodynamic Modelling

**Deltares — D-Flow FM User Manual**

Official documentation for **D-Flow Flexible Mesh hydrodynamic and flood-flow modelling**, covering model setup, boundary conditions and hydraulic simulation.

**Link:**
https://content.oss.deltares.nl/delft3d/D-Flow_FM_User_Manual.pdf

---

## 2. Uncertainty-Aware Flood Surrogates

**Siripatana, Wilson & Beevers (2025)**
*“Uncertainty Quantification for Multi-Input Fluvial Flood Inundation Using GPR- and PCE-Based Surrogates”*

Investigates faster flood-surrogate models while accounting for uncertainty in model inputs, supporting uncertainty-aware surrogate modelling.

**Link:**
https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024WR039668

---

## 3. Empirical Dam-Breach Models

**Catterick, Iliadis & Glenis (2026)**
*“A Review of Empirical Dam Breach Models”*

Reviews empirical dam-breach models and highlights uncertainty in breach size, formation time and peak outflow, supporting careful breach-scenario selection.

**Link:**
https://onlinelibrary.wiley.com/doi/10.1002/rvr2.70065

---

## 4. DeepONet for Dam-Break Hydrodynamics

**Gu & Lai (2026)**
*“Comparative Performance Evaluation of DeepONet Architectures for Dam-Break Hydrodynamic Simulations”*

Compares DeepONet architectures for dam-break flow prediction and demonstrates the potential of physics-informed surrogate modelling for faster hydrodynamic simulation.

**Link:**
https://www.sciencedirect.com/science/article/pii/S1674237026000049

---

# 🏆 Smart India Hackathon 2026

**Competition:** Smart India Hackathon 2026
**Problem Statement:** PS-161
**Theme:** Dam-Break Inundation Modelling Using Hydrodynamic Modelling of any River

### JalRaksha Approach

> **Hydrodynamics + AI Surrogate + Trust Assessment + GIS + Adaptive Evacuation + Digital Twin**

JalRaksha focuses on connecting **physics, AI and emergency decision support** rather than treating AI as an isolated black-box prediction system.

---

# 🔭 Vision

The long-term vision of JalRaksha is to evolve into a **validated, scalable and uncertainty-aware flood decision-support platform** capable of connecting:

```text
Hydraulic Simulation
        +
AI Surrogates
        +
Trust / Uncertainty
        +
Geospatial Intelligence
        +
Risk Assessment
        +
Evacuation Optimization
        ↓
Faster & More Explainable Flood Decisions
```

---

# 🛡️ Responsible Use

JalRaksha is a **research and prototype decision-support platform** intended for development, demonstration and experimentation.

Real-world emergency deployment requires appropriate validation, authoritative datasets, calibration and operational approval.

---

# 👥 Project

### JalRaksha — Smart India Hackathon 2026

**Hydrodynamics • AI • GIS • Trust • Risk • Evacuation • Visualization**

---

## ⭐ Core Idea

> **JalRaksha does not simply ask “What does the model predict?”**
>
> **It also asks “Can we trust this prediction for this scenario, what evidence supports it, and how should that uncertainty influence the decision?”**
