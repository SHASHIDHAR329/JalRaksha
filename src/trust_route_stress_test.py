from pathlib import Path
import csv

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# JalRaksha - Member 1
# Trust-Aware Adaptive Evacuation Routing
#
# CONTROLLED ROUTING STRESS TEST
#
# Uses the existing FNO flood field as the background.
# Creates three hypothetical road corridors:
#
#   Corridor A = shortest, but highly uncertain
#   Corridor B = medium detour, medium uncertainty
#   Corridor C = longest, low uncertainty
#
# The purpose is to prove that route selection changes
# when trust decreases.
#
# This is NOT GIS validation and NOT a real road network.
# ============================================================


PROJECT_ROOT = Path(r"C:\JalRaksha")

DATA_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "fno_dataset"
    / "Y.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "routing_stress_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_FILE = OUTPUT_DIR / "stress_test_results.csv"
PNG_FILE = OUTPUT_DIR / "stress_test_routes.png"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SCENARIO_INDEX = 1       # Scenario 02

START_COL = 5
GOAL_COL = 120

CORRIDOR_ROWS = {
    "Corridor A - shortest / uncertain": 2,
    "Corridor B - medium detour": 7,
    "Corridor C - longest / trusted": 12,
}

# Larger number = more uncertainty.
CORRIDOR_UNCERTAINTY = {
    "Corridor A - shortest / uncertain": 0.95,
    "Corridor B - medium detour": 0.35,
    "Corridor C - longest / trusted": 0.05,
}

TRUST_LEVELS = [
    100,
    75,
    50,
    25,
]

CELL_LENGTH_M = 1.0


# ------------------------------------------------------------
# Load existing FNO field
# ------------------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"FNO flood data not found:\n{DATA_FILE}"
    )

Y = np.load(DATA_FILE)

if Y.ndim != 4:
    raise ValueError(
        f"Expected Y shape (N,1,H,W), got {Y.shape}"
    )

depth = Y[
    SCENARIO_INDEX,
    0
].astype(float)

rows, cols = depth.shape

if GOAL_COL >= cols:
    raise ValueError(
        f"GOAL_COL={GOAL_COL} exceeds flood-grid width {cols}"
    )


print("=" * 72)
print("JalRaksha Trust-Aware Evacuation Routing Stress Test")
print("=" * 72)

print(f"Scenario: {SCENARIO_INDEX + 1}")
print(f"Flood field shape: {depth.shape}")
print(f"Flood maximum: {depth.max():.6f} m")
print(f"Flood mean:    {depth.mean():.6f} m")

print("\nControlled corridors:")

for name, row in CORRIDOR_ROWS.items():

    print(
        f"  {name}: row={row}, "
        f"uncertainty={CORRIDOR_UNCERTAINTY[name]:.2f}"
    )

print("=" * 72)


# ------------------------------------------------------------
# Build corridor graph
# ------------------------------------------------------------

G_BASE = nx.Graph()


def add_corridor(
    graph,
    name,
    row,
    uncertainty,
):
    """
    Add a horizontal road corridor.
    """

    # Horizontal section
    for col in range(
        START_COL,
        GOAL_COL + 1
    ):

        node = (row, col)

        flood_depth = float(
            depth[row, col]
        )

        graph.add_node(
            node,
            corridor=name,
            flood_depth_m=flood_depth,
            uncertainty=uncertainty,
        )

        if col > START_COL:

            prev = (row, col - 1)

            graph.add_edge(
                prev,
                node,
                length=CELL_LENGTH_M,
                corridor=name,
                uncertainty=uncertainty,
            )


def add_connector(
    graph,
    top_row,
    bottom_row,
    col,
):
    """
    Connect two corridors vertically.
    """

    step = 1 if bottom_row >= top_row else -1

    current_row = top_row

    while current_row != bottom_row:

        next_row = current_row + step

        a = (current_row, col)
        b = (next_row, col)

        # Make sure nodes exist.
        for node in (a, b):

            if node not in graph:

                r, c = node

                graph.add_node(
                    node,
                    corridor="connector",
                    flood_depth_m=float(depth[r, c]),
                    uncertainty=0.20,
                )

        graph.add_edge(
            a,
            b,
            length=CELL_LENGTH_M,
            corridor="connector",
            uncertainty=0.20,
        )

        current_row = next_row


# Add three corridors.
for name, row in CORRIDOR_ROWS.items():

    add_corridor(
        G_BASE,
        name,
        row,
        CORRIDOR_UNCERTAINTY[name],
    )


# Connect all corridors at beginning and end.
corridor_row_list = list(CORRIDOR_ROWS.values())

for col in (START_COL, GOAL_COL):

    for i in range(len(corridor_row_list) - 1):

        add_connector(
            G_BASE,
            corridor_row_list[i],
            corridor_row_list[i + 1],
            col,
        )


START = (
    CORRIDOR_ROWS[
        "Corridor A - shortest / uncertain"
    ],
    START_COL,
)

GOAL = (
    CORRIDOR_ROWS[
        "Corridor A - shortest / uncertain"
    ],
    GOAL_COL,
)


# ------------------------------------------------------------
# Routing cost
# ------------------------------------------------------------

def calculate_edge_cost(
    edge_data,
    trust_score,
):
    """
    Compute the adaptive routing cost.

    Trust = confidence in the flood prediction.

    Lower trust increases the cost of uncertain roads.
    """

    length = edge_data["length"]
    uncertainty = edge_data["uncertainty"]

    # Trust converted from 0-100 to 0-1.
    trust_fraction = trust_score / 100.0

    # Remaining uncertainty.
    uncertainty_exposure = (
        1.0 - trust_fraction
    )

    # Stronger penalty for uncertain corridors.
    trust_penalty = (
        1.0
        + 8.0
        * uncertainty
        * uncertainty_exposure
    )

    return length * trust_penalty


def build_trust_graph(
    trust_score,
):
    """
    Build a graph whose edge weights respond
    to the current trust level.
    """

    G = G_BASE.copy()

    for u, v, data in G.edges(data=True):

        data["routing_cost"] = calculate_edge_cost(
            data,
            trust_score,
        )

    return G


# ------------------------------------------------------------
# Determine which corridor carries the route
# ------------------------------------------------------------

def identify_main_corridor(path):

    corridor_counts = {}

    for node in path:

        if node not in G_BASE.nodes:
            continue

        corridor = G_BASE.nodes[node].get(
            "corridor",
            "unknown"
        )

        if corridor == "connector":
            continue

        corridor_counts[corridor] = (
            corridor_counts.get(corridor, 0)
            + 1
        )

    if not corridor_counts:
        return "unknown"

    return max(
        corridor_counts,
        key=corridor_counts.get
    )


# ------------------------------------------------------------
# Run experiments
# ------------------------------------------------------------

results = []
saved_routes = {}


for trust in TRUST_LEVELS:

    G = build_trust_graph(trust)

    try:

        path = nx.shortest_path(
            G,
            source=START,
            target=GOAL,
            weight="routing_cost",
        )

    except nx.NetworkXNoPath:

        print(
            f"\nTrust {trust}: NO ROUTE"
        )

        continue


    physical_distance = 0.0
    routing_cost = 0.0
    path_depths = []

    for a, b in zip(
        path[:-1],
        path[1:],
    ):

        edge = G[a][b]

        physical_distance += edge[
            "length"
        ]

        routing_cost += edge[
            "routing_cost"
        ]


    for node in path:

        if node in G.nodes:

            path_depths.append(
                G.nodes[node][
                    "flood_depth_m"
                ]
            )


    main_corridor = identify_main_corridor(
        path
    )

    max_depth = max(path_depths)
    mean_depth = float(
        np.mean(path_depths)
    )


    print()
    print("-" * 72)
    print(
        f"Trust level: {trust}/100"
    )
    print(
        f"Selected corridor: {main_corridor}"
    )
    print(
        f"Physical distance: "
        f"{physical_distance:.2f} m"
    )
    print(
        f"Adaptive routing cost: "
        f"{routing_cost:.2f}"
    )
    print(
        f"Maximum flood depth: "
        f"{max_depth:.6f} m"
    )
    print(
        f"Mean flood depth: "
        f"{mean_depth:.6f} m"
    )
    print(
        f"Route nodes: {len(path)}"
    )


    results.append({

        "trust_score": trust,

        "selected_corridor":
            main_corridor,

        "physical_distance_m":
            physical_distance,

        "adaptive_routing_cost":
            routing_cost,

        "max_depth_m":
            max_depth,

        "mean_depth_m":
            mean_depth,

        "route_nodes":
            len(path),
    })


    saved_routes[trust] = path


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "trust_score",
        "selected_corridor",
        "physical_distance_m",
        "adaptive_routing_cost",
        "max_depth_m",
        "mean_depth_m",
        "route_nodes",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(results)


# ------------------------------------------------------------
# Visualize
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(16, 7)
)

# Flood background.
im = ax.imshow(
    depth,
    origin="upper",
    aspect="auto",
)

fig.colorbar(
    im,
    ax=ax,
    label="FNO predicted flood depth (m)",
)


# Plot corridor locations.
for name, row in CORRIDOR_ROWS.items():

    ax.plot(
        [START_COL, GOAL_COL],
        [row, row],
        linestyle="--",
        linewidth=1,
        alpha=0.35,
    )

    ax.text(
        GOAL_COL + 1,
        row,
        name,
        va="center",
        fontsize=8,
    )


# Start and destination.
ax.scatter(
    [START_COL],
    [CORRIDOR_ROWS[
        "Corridor A - shortest / uncertain"
    ]],
    s=120,
    marker="o",
    label="Evacuation start",
)

ax.scatter(
    [GOAL_COL],
    [CORRIDOR_ROWS[
        "Corridor A - shortest / uncertain"
    ]],
    s=160,
    marker="*",
    label="Destination",
)


# Draw routes.
for trust, path in saved_routes.items():

    x = [
        node[1]
        for node in path
    ]

    y = [
        node[0]
        for node in path
    ]

    ax.plot(
        x,
        y,
        linewidth=3,
        label=f"Trust {trust}",
    )


ax.set_title(
    "JalRaksha — Trust-Aware Adaptive "
    "Evacuation Routing Stress Test"
)

ax.set_xlabel(
    "Flood grid column"
)

ax.set_ylabel(
    "Flood grid row"
)

ax.legend(
    loc="upper left"
)

fig.tight_layout()

fig.savefig(
    PNG_FILE,
    dpi=200,
)

plt.close(fig)


# ------------------------------------------------------------
# Final message
# ------------------------------------------------------------

print()
print("=" * 72)
print("STRESS TEST COMPLETE")
print("=" * 72)

print(
    "CSV:"
)
print(
    CSV_FILE
)

print()
print(
    "Visualization:"
)
print(
    PNG_FILE
)

print()
print(
    "Purpose:"
)
print(
    "Demonstrate that decreasing trust can make "
    "the router choose a longer, less uncertain "
    "evacuation corridor."
)

print()
print(
    "Important:"
)
print(
    "This is a controlled algorithm stress test. "
    "The corridor uncertainty values are synthetic "
    "demo parameters, not measured road uncertainty."
)

print("=" * 72)