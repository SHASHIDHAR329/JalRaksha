from pathlib import Path
import csv
import math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# JalRaksha - Member 1
# Trust-Aware Adaptive Evacuation Routing
#
# Standalone prototype:
#   Existing flood depth field
#        ↓
#   Trust-aware road cost
#        ↓
#   Adaptive route
#
# This version intentionally uses the existing FNO grid rather
# than requiring OSM/GIS/Member-3 inputs.
# ============================================================


PROJECT_ROOT = Path(r"C:\JalRaksha")

DATA_FILE = PROJECT_ROOT / "outputs" / "fno_dataset" / "Y.npy"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "routing_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "trust_route_comparison.csv"
PNG_FILE = OUTPUT_DIR / "trust_route_comparison.png"


# ------------------------------------------------------------
# Demo configuration
# ------------------------------------------------------------

SCENARIO_INDEX = 1       # scenario_02
START = (2, 5)           # row, col
GOAL = (13, 120)         # row, col

CELL_LENGTH_M = 1.0

# Engineering-demo threshold only.
# This is NOT a regulatory evacuation threshold.
BLOCK_DEPTH_M = 0.225

# Depth above this value receives increasing penalty.
REFERENCE_DEPTH_M = 0.20


# ------------------------------------------------------------
# Load flood depth
# ------------------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Flood dataset not found:\n{DATA_FILE}"
    )

Y = np.load(DATA_FILE)

if Y.ndim != 4:
    raise ValueError(
        f"Expected Y shape (N, 1, H, W), got {Y.shape}"
    )

depth = Y[SCENARIO_INDEX, 0].astype(float)

rows, cols = depth.shape

print("=" * 70)
print("JalRaksha Trust-Aware Adaptive Evacuation Routing")
print("=" * 70)
print(f"Dataset: {DATA_FILE}")
print(f"Scenario index: {SCENARIO_INDEX + 1}")
print(f"Flood grid: {rows} x {cols}")
print(f"Start: {START}")
print(f"Goal:  {GOAL}")
print(f"Max depth:  {depth.max():.6f} m")
print(f"Mean depth: {depth.mean():.6f} m")
print("=" * 70)


# ------------------------------------------------------------
# Validate start / goal
# ------------------------------------------------------------

def valid_cell(cell):
    r, c = cell
    return (
        0 <= r < rows
        and 0 <= c < cols
        and np.isfinite(depth[r, c])
    )


if not valid_cell(START):
    raise ValueError(f"Invalid START cell: {START}")

if not valid_cell(GOAL):
    raise ValueError(f"Invalid GOAL cell: {GOAL}")


# ------------------------------------------------------------
# Build grid road graph
# ------------------------------------------------------------

def build_graph():
    G = nx.Graph()

    for r in range(rows):
        for c in range(cols):

            if not np.isfinite(depth[r, c]):
                continue

            node = (r, c)

            G.add_node(
                node,
                flood_depth_m=float(depth[r, c])
            )

            # Right neighbour
            if c + 1 < cols and np.isfinite(depth[r, c + 1]):
                nbr = (r, c + 1)

                G.add_edge(
                    node,
                    nbr,
                    length=CELL_LENGTH_M
                )

            # Down neighbour
            if r + 1 < rows and np.isfinite(depth[r + 1, c]):
                nbr = (r + 1, c)

                G.add_edge(
                    node,
                    nbr,
                    length=CELL_LENGTH_M
                )

    return G


BASE_GRAPH = build_graph()

print(f"Road/grid nodes: {BASE_GRAPH.number_of_nodes()}")
print(f"Road/grid edges: {BASE_GRAPH.number_of_edges()}")


# ------------------------------------------------------------
# Trust-aware routing
# ------------------------------------------------------------

def route_graph(
    flood_trust_score,
    block_depth_m=BLOCK_DEPTH_M
):

    G = BASE_GRAPH.copy()

    trust = max(0.0, min(100.0, float(flood_trust_score)))

    # Lower trust means greater caution around uncertain flood cells.
    uncertainty_factor = (100.0 - trust) / 100.0

    blocked = 0

    for u, v, data in list(G.edges(data=True)):

        du = G.nodes[u]["flood_depth_m"]
        dv = G.nodes[v]["flood_depth_m"]

        edge_max_depth = max(du, dv)
        edge_mean_depth = (du + dv) / 2.0

        # ----------------------------------------------------
        # Hard blocking:
        # Deeply flooded road segment is considered unusable.
        # ----------------------------------------------------
        if edge_max_depth >= block_depth_m:
            G.remove_edge(u, v)
            blocked += 1
            continue

        # ----------------------------------------------------
        # Flood penalty
        # ----------------------------------------------------

        excess_depth = max(
            0.0,
            edge_mean_depth - REFERENCE_DEPTH_M
        )

        depth_penalty = 1.0 + 40.0 * excess_depth

        # ----------------------------------------------------
        # Trust penalty
        #
        # Lower trust -> more conservative routing.
        # ----------------------------------------------------

        uncertainty_penalty = (
            1.0 +
            uncertainty_factor *
            (1.0 + 80.0 * excess_depth)
        )

        routing_cost = (
            data["length"]
            * depth_penalty
            * uncertainty_penalty
        )

        data["routing_cost"] = routing_cost
        data["flood_max_depth_m"] = edge_max_depth
        data["flood_mean_depth_m"] = edge_mean_depth

    return G, blocked


# ------------------------------------------------------------
# Evaluate a route
# ------------------------------------------------------------

def evaluate_route(G, path):

    if not path:
        return None

    depths = [
        G.nodes[n]["flood_depth_m"]
        for n in path
    ]

    total_distance = 0.0
    max_depth = max(depths)
    mean_depth = float(np.mean(depths))

    for a, b in zip(path[:-1], path[1:]):
        total_distance += G[a][b]["length"]

    return {
        "distance_m": total_distance,
        "max_depth_m": max_depth,
        "mean_depth_m": mean_depth,
        "cells": len(path),
    }


# ------------------------------------------------------------
# Run scenarios
# ------------------------------------------------------------

experiments = [
    ("Shortest / no flood intelligence", None),
    ("Trust 100 / high confidence", 100),
    ("Trust 75 / moderate confidence", 75),
    ("Trust 50 / cautious", 50),
]


results = []
routes = {}


for label, trust in experiments:

    if trust is None:

        G = BASE_GRAPH.copy()

        for u, v, data in G.edges(data=True):
            data["routing_cost"] = data["length"]

        blocked = 0

    else:

        G, blocked = route_graph(trust)

    try:

        path = nx.shortest_path(
            G,
            source=START,
            target=GOAL,
            weight="routing_cost"
        )

    except nx.NetworkXNoPath:

        print(f"\nNO ROUTE: {label}")

        results.append({
            "mode": label,
            "trust_score": "" if trust is None else trust,
            "route_available": False,
            "distance_m": "",
            "max_depth_m": "",
            "mean_depth_m": "",
            "route_cells": "",
            "blocked_edges": blocked
        })

        routes[label] = None
        continue

    evaluation = evaluate_route(G, path)

    results.append({
        "mode": label,
        "trust_score": "" if trust is None else trust,
        "route_available": True,
        "distance_m": evaluation["distance_m"],
        "max_depth_m": evaluation["max_depth_m"],
        "mean_depth_m": evaluation["mean_depth_m"],
        "route_cells": evaluation["cells"],
        "blocked_edges": blocked
    })

    routes[label] = path

    print(f"\n{label}")
    print(f"  Trust score:   {trust}")
    print(f"  Route cells:   {evaluation['cells']}")
    print(f"  Distance:      {evaluation['distance_m']:.2f} m")
    print(f"  Max depth:     {evaluation['max_depth_m']:.6f} m")
    print(f"  Mean depth:    {evaluation['mean_depth_m']:.6f} m")
    print(f"  Blocked edges: {blocked}")


# ------------------------------------------------------------
# Save comparison CSV
# ------------------------------------------------------------

with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:

    fieldnames = [
        "mode",
        "trust_score",
        "route_available",
        "distance_m",
        "max_depth_m",
        "mean_depth_m",
        "route_cells",
        "blocked_edges",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in results:
        writer.writerow(row)


# ------------------------------------------------------------
# Visual comparison
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(16, 6))

ax.imshow(
    depth,
    origin="upper"
)

ax.scatter(
    [START[1]],
    [START[0]],
    marker="o",
    s=100,
    label="Evacuation start"
)

ax.scatter(
    [GOAL[1]],
    [GOAL[0]],
    marker="*",
    s=160,
    label="Safe destination"
)


for label, path in routes.items():

    if path is None:
        continue

    y = [p[0] for p in path]
    x = [p[1] for p in path]

    ax.plot(
        x,
        y,
        linewidth=2,
        label=label
    )


ax.set_title(
    "JalRaksha — Trust-Aware Adaptive Evacuation Routing"
)

ax.set_xlabel("Flood grid column")
ax.set_ylabel("Flood grid row")

ax.legend(
    loc="upper left",
    fontsize=8
)

fig.tight_layout()

fig.savefig(
    PNG_FILE,
    dpi=200
)

plt.close(fig)


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ROUTING DEMO COMPLETE")
print("=" * 70)

print(f"CSV output:")
print(CSV_FILE)

print(f"\nRoute visualization:")
print(PNG_FILE)

print("\nThis experiment demonstrates:")
print("1. Flood depth changes routing cost.")
print("2. Deep segments can become blocked.")
print("3. Trust changes routing conservativeness.")
print("4. Different trust levels can produce different routes.")

print("\nImportant:")
print(
    "The blocking threshold in this experiment is an engineering "
    "demo parameter, not an official safety threshold."
)

print("=" * 70)