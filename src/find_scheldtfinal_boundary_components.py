from pathlib import Path
import numpy as np
import xarray as xr

NET_FILE = Path(
    r"C:\JalRaksha\simulations\dflowfm\tutorials"
    r"\Tutorial_RGFGRID\fm_scheldt\finished\scheldtfinal_net.nc"
)

ds = xr.open_dataset(NET_FILE)

netlink = ds["NetLink"].values
bnd_links = ds["BndLink"].values
x = ds["NetNode_x"].values
y = ds["NetNode_y"].values

# Convert boundary links to endpoint-node pairs
edges = []

for bnd_pos, link_number in enumerate(bnd_links):
    link_idx = int(link_number) - 1

    n1 = int(netlink[link_idx, 0]) - 1
    n2 = int(netlink[link_idx, 1]) - 1

    edges.append((bnd_pos, int(link_number), n1, n2))

# Build node -> boundary-edge map
node_edges = {}

for edge_idx, (_, _, n1, n2) in enumerate(edges):
    node_edges.setdefault(n1, []).append(edge_idx)
    node_edges.setdefault(n2, []).append(edge_idx)

# Find connected components of boundary edges
visited = set()
components = []

for start in range(len(edges)):
    if start in visited:
        continue

    stack = [start]
    visited.add(start)
    component = []

    while stack:
        current = stack.pop()
        component.append(current)

        _, _, n1, n2 = edges[current]

        for node in (n1, n2):
            for next_edge in node_edges[node]:
                if next_edge not in visited:
                    visited.add(next_edge)
                    stack.append(next_edge)

    components.append(component)

# Sort largest first
components.sort(key=len, reverse=True)

print("Total boundary links:", len(edges))
print("Connected boundary components:", len(components))

print("\n=== COMPONENT SUMMARY ===")

for i, component in enumerate(components):
    nodes = set()

    for edge_idx in component:
        _, _, n1, n2 = edges[edge_idx]
        nodes.add(n1)
        nodes.add(n2)

    xs = x[list(nodes)]
    ys = y[list(nodes)]

    print(
        f"Component {i + 1:2d}: "
        f"{len(component):4d} links | "
        f"{len(nodes):4d} nodes | "
        f"X {xs.min():.1f} to {xs.max():.1f} | "
        f"Y {ys.min():.1f} to {ys.max():.1f}"
    )

print("\n=== LARGEST COMPONENTS: FIRST/LAST LINKS ===")

for i, component in enumerate(components[:10]):
    first = edges[component[0]]
    last = edges[component[-1]]

    print(
        f"\nComponent {i + 1}: {len(component)} links"
    )

    print(
        f"  First: Bnd {first[0]}, Link {first[1]}"
    )
    print(
        f"         ({x[first[2]]:.2f}, {y[first[2]]:.2f}) -> "
        f"({x[first[3]]:.2f}, {y[first[3]]:.2f})"
    )

    print(
        f"  Last : Bnd {last[0]}, Link {last[1]}"
    )
    print(
        f"         ({x[last[2]]:.2f}, {y[last[2]]:.2f}) -> "
        f"({x[last[3]]:.2f}, {y[last[3]]:.2f})"
    )