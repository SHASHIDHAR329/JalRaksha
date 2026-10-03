import rasterio
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

depth_file = "gis/output/test_flood_depth.tif"
extent_file = "gis/output/pipeline_flood_extent.tif"
hazard_file = "gis/output/pipeline_hazard.tif"
impact_file = "gis/output/pipeline_impact.tif"

output_file = "gis/output/gis_layers_overview.png"


# ---------------------------------------------------------
# READ FLOOD DEPTH
# ---------------------------------------------------------

with rasterio.open(depth_file) as src:
    depth = src.read(1)
    bounds = src.bounds


# ---------------------------------------------------------
# READ FLOOD EXTENT
# ---------------------------------------------------------

with rasterio.open(extent_file) as src:
    extent = src.read(1)


# ---------------------------------------------------------
# READ HAZARD
# ---------------------------------------------------------

with rasterio.open(hazard_file) as src:
    hazard = src.read(1)


# ---------------------------------------------------------
# READ IMPACT
# ---------------------------------------------------------

with rasterio.open(impact_file) as src:
    impact = src.read(1)


# ---------------------------------------------------------
# CREATE FIGURE
# ---------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))


# ---------------------------------------------------------
# FLOOD DEPTH
# ---------------------------------------------------------

image1 = axes[0, 0].imshow(
    depth,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ],
    origin="upper"
)

axes[0, 0].set_title("Flood Depth")
axes[0, 0].set_xlabel("Longitude")
axes[0, 0].set_ylabel("Latitude")

fig.colorbar(
    image1,
    ax=axes[0, 0],
    label="Depth (m)"
)


# ---------------------------------------------------------
# FLOOD EXTENT
# ---------------------------------------------------------

image2 = axes[0, 1].imshow(
    extent,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ],
    origin="upper"
)

axes[0, 1].set_title("Flood Extent")
axes[0, 1].set_xlabel("Longitude")
axes[0, 1].set_ylabel("Latitude")

fig.colorbar(
    image2,
    ax=axes[0, 1],
    label="Flooded (0/1)"
)


# ---------------------------------------------------------
# HAZARD
# ---------------------------------------------------------

image3 = axes[1, 0].imshow(
    hazard,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ],
    origin="upper"
)

axes[1, 0].set_title(
    "Hazard Classification"
)

axes[1, 0].set_xlabel("Longitude")
axes[1, 0].set_ylabel("Latitude")

fig.colorbar(
    image3,
    ax=axes[1, 0],
    label="Hazard Class"
)


# ---------------------------------------------------------
# IMPACT
# ---------------------------------------------------------

image4 = axes[1, 1].imshow(
    impact,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ],
    origin="upper"
)

axes[1, 1].set_title(
    "Impact Map"
)

axes[1, 1].set_xlabel("Longitude")
axes[1, 1].set_ylabel("Latitude")

fig.colorbar(
    image4,
    ax=axes[1, 1],
    label="Impact Class"
)


# ---------------------------------------------------------
# FINAL LAYOUT
# ---------------------------------------------------------

fig.suptitle(
    "JalRaksha Flood GIS Analysis",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

print("GIS overview map created successfully.")
print(f"Output: {output_file}")

plt.show()