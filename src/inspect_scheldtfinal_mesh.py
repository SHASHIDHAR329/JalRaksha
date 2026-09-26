from pathlib import Path
import xarray as xr

NET_FILE = Path(
    r"C:\JalRaksha\simulations\dflowfm\tutorials"
    r"\Tutorial_RGFGRID\fm_scheldt\finished\scheldtfinal_net.nc"
)

ds = xr.open_dataset(NET_FILE)

print("=== DIMENSIONS ===")
for name, size in ds.sizes.items():
    print(f"{name}: {size}")

print("\n=== VARIABLES ===")
for name in ds.variables:
    print(name)

print("\n=== BOUNDARY-RELATED VARIABLES ===")
for name in ds.variables:
    lower = name.lower()
    if any(word in lower for word in ["bnd", "boundary", "netlink", "coord", "node", "elem"]):
        print(name)

print("\n=== GLOBAL ATTRIBUTES ===")
for key, value in ds.attrs.items():
    print(f"{key}: {value}")