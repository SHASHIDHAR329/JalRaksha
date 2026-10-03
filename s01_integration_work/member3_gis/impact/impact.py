import numpy as np


def population_exposed(flood_depth, population_grid, threshold=0.1):
    """
    Calculate the population located in flooded cells.

    flood_depth:
        Flood depth array.

    population_grid:
        Population associated with each grid cell.

    threshold:
        Minimum flood depth considered as flooding.
    """

    flooded = flood_depth >= threshold

    return int(np.sum(population_grid[flooded]))


def assets_exposed(flood_depth, asset_grid, threshold=0.1):
    """
    Calculate the number of assets located in flooded cells.

    flood_depth:
        Flood depth array.

    asset_grid:
        Number of assets associated with each grid cell.

    threshold:
        Minimum flood depth considered as flooding.
    """

    flooded = flood_depth >= threshold

    return int(np.sum(asset_grid[flooded]))


def flood_area(flood_depth, cell_area_m2, threshold=0.1):
    """
    Calculate total flooded area.

    Returns area in square metres.
    """

    flooded = flood_depth >= threshold

    return float(np.sum(flooded) * cell_area_m2)