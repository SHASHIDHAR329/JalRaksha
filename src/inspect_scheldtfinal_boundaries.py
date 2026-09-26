from pathlib import Path
import xarray as xr
import numpy as np

NET_FILE = Path(
    r"C:\JalRaksha\simulations\dflowfm\tutorials"
    r"\\Tutorial_RGFGRID\fm_scheldt\finished\scheldtfinal_net.nc"
)

ds = xr.open_dataset(NET_FILE)

print("=== BOUNDARY ARRAY ===")
print("BndLink dimensions:", ds["BndLink"].dims)
print("BndLink shape:", ds["BndLink"].shape)
print("First 30 BndLink values:")
print(ds["BndLink"].values[:30])

print("\n=== NETLINK ARRAY ===")
print("NetLink dimensions:", ds["NetLink"].dims)
print("NetLink shape:", ds["NetLink"].shape)
print("First 10 NetLink rows:")
print(ds["NetLink"].values[:10])

print("\n=== NODE COORDINATE RANGE ===")

x = ds["NetNode_x"].values
y = ds["NetNode_y"].values

print(f"X min: {np.min(x):.3f}")
print(f"X max: {np.max(x):.3f}")
print(f"Y min: {np.min(y):.3f}")
print(f"Y max: {np.max(y):.3f}")

print("\n=== BOUNDARY LINK ENDPOINTS ===")

bnd = np.asarray(ds["BndLink"].values)

# Flatten in case BndLink has an extra dimension
bnd = bnd.reshape(-1)

netlink = np.asarray(ds["NetLink"].values)

for i, link_id in enumerate(bnd[:50]):
    try:
        link_id = int(link_id)

        if link_id < 0:
            link_id = abs(link_id)

        nodes = netlink[link_id]

        n1 = int(nodes[0])
        n2 = int(nodes[1])

        print(
            f"Bnd {i:3d}: "
            f"Link {link_id:5d} | "
            f"({x[n1]:.3f}, {y[n1]:.3f}) -> "
            f"({x[n2]:.3f}, {y[n2]:.3f})"
        )

    except Exception as e:
        print(f"Bnd {i}: could not decode ({e})")