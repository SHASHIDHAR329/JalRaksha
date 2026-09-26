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
link_type = ds["NetLinkType"].values

print("Total boundary links:", len(bnd_links))

# BndLink/NetLink use node/link numbering from the mesh file,
# so convert to Python zero-based indices.
boundary_records = []

for bnd_pos, link_number in enumerate(bnd_links):
    link_idx = int(link_number) - 1

    if link_idx < 0 or link_idx >= len(netlink):
        continue

    n1 = int(netlink[link_idx, 0]) - 1
    n2 = int(netlink[link_idx, 1]) - 1

    if n1 < 0 or n2 < 0 or n1 >= len(x) or n2 >= len(x):
        continue

    boundary_records.append({
        "bnd_pos": bnd_pos,
        "link": int(link_number),
        "n1": n1,
        "n2": n2,
        "x1": x[n1],
        "y1": y[n1],
        "x2": x[n2],
        "y2": y[n2],
        "type": link_type[link_idx]
    })

print("\n=== BOUNDARY LINK-TYPE COUNTS ===")

types = {}

for r in boundary_records:
    key = str(r["type"])
    types[key] = types.get(key, 0) + 1

for key, count in types.items():
    print(f"Type {key}: {count} links")

print("\n=== BOUNDARY OVERALL EXTENT ===")

all_x = []
all_y = []

for r in boundary_records:
    all_x.extend([r["x1"], r["x2"]])
    all_y.extend([r["y1"], r["y2"]])

print(f"X: {min(all_x):.3f} to {max(all_x):.3f}")
print(f"Y: {min(all_y):.3f} to {max(all_y):.3f}")

print("\n=== SAMPLE BOUNDARY SECTIONS ===")

# Print every 20th boundary link so we can see how the boundary
# travels around the domain without flooding the terminal.
for i in range(0, len(boundary_records), 20):
    r = boundary_records[i]

    print(
        f"Bnd {r['bnd_pos']:3d} | "
        f"Link {r['link']:4d} | "
        f"Type {r['type']} | "
        f"({r['x1']:.2f}, {r['y1']:.2f}) -> "
        f"({r['x2']:.2f}, {r['y2']:.2f})"
    )

print("\n=== LAST 20 BOUNDARY LINKS ===")

for r in boundary_records[-20:]:
    print(
        f"Bnd {r['bnd_pos']:3d} | "
        f"Link {r['link']:4d} | "
        f"Type {r['type']} | "
        f"({r['x1']:.2f}, {r['y1']:.2f}) -> "
        f"({r['x2']:.2f}, {r['y2']:.2f})"
    )