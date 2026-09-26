JALRAKSHA SAR VALIDATION INPUTS

Required files:

1. fno_flood_mask.tif
   Binary georeferenced flood mask
   1 = flooded
   0 = non-flooded

2. sentinel1_flood_mask.tif
   Binary georeferenced Sentinel-1-derived flood mask
   1 = flooded
   0 = non-flooded

IMPORTANT:
Both rasters must have:
- identical CRS
- identical pixel dimensions
- identical geotransform/grid
- matching spatial extent

The current DualSPHysics/D-Flow FM flume case uses local laboratory
coordinates, so it must not be presented as a real satellite location.
