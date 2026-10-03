from pathlib import Path
import math
import json

import numpy as np
import osmnx as ox
import networkx as nx
import rasterio
from rasterio.warp import transform


ROOT = Path(r"C:\JalRaksha")

ROUTING_DIR = ROOT / r"outputs\evacuation"
ROUTING_DIR.mkdir(parents=True, exist_ok=True)


def load_flood_raster(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Flood raster not found:\n{path}"
        )

    src = rasterio.open(path)

    depth = src.read(1).astype(np.float32)

    transform_obj = src.transform
    crs = src.crs
    nodata = src.nodata

    return src, depth, transform_obj, crs, nodata


def sample_raster(
    src,
    depth,
    points_xy,
    point_crs="EPSG:4326",
):

    if src.crs is None:
        raise ValueError(
            "Flood raster must have a valid CRS."
        )

    xs = np.array(
        [p[0] for p in points_xy],
        dtype=float
    )

    ys = np.array(
        [p[1] for p in points_xy],
        dtype=float
    )

    if point_crs != str(src.crs):

        tx, ty = transform(
            point_crs,
            src.crs,
            xs.tolist(),
            ys.tolist(),
        )

        xs = np.asarray(tx)
        ys = np.asarray(ty)

    values = []

    for x, y in zip(xs, ys):

        try:
            row, col = src.index(
                x,
                y
            )

            if (
                0 <= row < depth.shape[0]
                and 0 <= col < depth.shape[1]
            ):
                value = float(
                    depth[row, col]
                )

                if (
                    src.nodata is not None
                    and value == src.nodata
                ):
                    value = 0.0

            else:
                value = 0.0

        except Exception:
            value = 0.0

        values.append(value)

    return np.asarray(
        values,
        dtype=float
    )


def edge_risk_cost(
    geometry,
    flood_src,
    flood_depth,
    trust_score,
    sample_count=9,
    block_depth_m=0.30,
    penalty_strength=8.0,
):

    if geometry is None:
        return None

    length = float(
        geometry.length
    )

    if length <= 0:
        return None

    distances = np.linspace(
        0,
        length,
        sample_count
    )

    points = [
        geometry.interpolate(
            float(distance)
        )
        for distance in distances
    ]

    xy = [
        (
            point.x,
            point.y
        )
        for point in points
    ]

    depths = sample_raster(
        flood_src,
        flood_depth,
        xy,
        point_crs=str(
            flood_src.crs
        ),
    )

    max_depth = float(
        np.max(depths)
    )

    mean_depth = float(
        np.mean(depths)
    )

    # --------------------------------------------------------
    # Hard block
    # --------------------------------------------------------

    if max_depth >= block_depth_m:

        return {
            "blocked": True,
            "cost": float("inf"),
            "length_m": length,
            "max_depth_m": max_depth,
            "mean_depth_m": mean_depth,
        }

    # --------------------------------------------------------
    # Trust-aware adjustment
    # --------------------------------------------------------

    trust = max(
        0.0,
        min(
            100.0,
            float(trust_score)
        )
    )

    # Lower trust -> more conservative routing.
    uncertainty_factor = (
        1.0
        + (100.0 - trust) / 100.0
    )

    depth_factor = (
        1.0
        + penalty_strength
        * uncertainty_factor
        * mean_depth
    )

    cost = (
        length
        * depth_factor
    )

    return {
        "blocked": False,
        "cost": float(cost),
        "length_m": length,
        "max_depth_m": max_depth,
        "mean_depth_m": mean_depth,
    }


def build_routing_graph(
    latitude,
    longitude,
    dist_meters=3000,
    network_type="walk",
):

    graph = ox.graph.graph_from_point(
        (
            latitude,
            longitude
        ),
        dist=dist_meters,
        network_type=network_type,
        simplify=True,
    )

    return graph


def assign_flood_risk(
    graph,
    flood_raster,
    trust_score,
    block_depth_m=0.30,
    penalty_strength=8.0,
):

    src, depth, _, _, _ = load_flood_raster(
        flood_raster
    )

    blocked_edges = 0

    for u, v, k, data in graph.edges(
        keys=True,
        data=True
    ):

        geometry = data.get(
            "geometry"
        )

        if geometry is None:
            geometry = None

        risk = edge_risk_cost(
            geometry,
            src,
            depth,
            trust_score,
            block_depth_m=block_depth_m,
            penalty_strength=penalty_strength,
        )

        if risk is None:

            data["routing_cost"] = float(
                data.get(
                    "length",
                    1.0
                )
            )

            data["flood_blocked"] = False
            data["flood_max_depth_m"] = 0.0
            data["flood_mean_depth_m"] = 0.0

            continue

        data["routing_cost"] = risk[
            "cost"
        ]

        data["flood_blocked"] = risk[
            "blocked"
        ]

        data["flood_max_depth_m"] = risk[
            "max_depth_m"
        ]

        data["flood_mean_depth_m"] = risk[
            "mean_depth_m"
        ]

        if risk["blocked"]:
            blocked_edges += 1

    src.close()

    return graph, blocked_edges


def route(
    graph,
    start_lat,
    start_lon,
    end_lat,
    end_lon,
):

    start_node = ox.distance.nearest_nodes(
        graph,
        X=start_lon,
        Y=start_lat,
    )

    end_node = ox.distance.nearest_nodes(
        graph,
        X=end_lon,
        Y=end_lat,
    )

    path = nx.shortest_path(
        graph,
        start_node,
        end_node,
        weight="routing_cost",
    )

    return path


def route_length(
    graph,
    path
):

    total = 0.0

    for u, v in zip(
        path[:-1],
        path[1:]
    ):

        edge_data = graph.get_edge_data(
            u,
            v
        )

        if edge_data is None:
            continue

        usable_edges = [
            d
            for d in edge_data.values()
            if not d.get(
                "flood_blocked",
                False
            )
        ]

        if not usable_edges:
            continue

        edge = min(
            usable_edges,
            key=lambda d: d.get(
                "routing_cost",
                d.get(
                    "length",
                    1.0
                )
            )
        )

        total += float(
            edge.get(
                "length",
                0.0
            )
        )

    return total


def save_route_geojson(
    graph,
    path,
    output_file
):

    nodes, edges = ox.convert.graph_to_gdfs(
        graph
    )

    route_edges = []

    for u, v in zip(
        path[:-1],
        path[1:]
    ):

        data = graph.get_edge_data(
            u,
            v
        )

        if data is None:
            continue

        candidates = [
            d
            for d in data.values()
            if d.get(
                "geometry"
            ) is not None
            and not d.get(
                "flood_blocked",
                False
            )
        ]

        if candidates:

            edge = min(
                candidates,
                key=lambda d: d.get(
                    "routing_cost",
                    d.get(
                        "length",
                        1.0
                    )
                )
            )

            route_edges.append(
                edge
            )

    if not route_edges:
        return False

    geoms = [
        edge["geometry"]
        for edge in route_edges
        if "geometry" in edge
    ]

    import geopandas as gpd

    route_gdf = gpd.GeoDataFrame(
        {
            "route_order": range(
                len(geoms)
            )
        },
        geometry=geoms,
        crs=edges.crs,
    )

    route_gdf.to_file(
        output_file,
        driver="GeoJSON"
    )

    return True


def create_evacuation_route(
    latitude,
    longitude,
    start_lat,
    start_lon,
    end_lat,
    end_lon,
    flood_raster,
    trust_score,
    output_name="evacuation_route",
    dist_meters=3000,
    network_type="walk",
    block_depth_m=0.30,
    penalty_strength=8.0,
):

    graph = build_routing_graph(
        latitude,
        longitude,
        dist_meters=dist_meters,
        network_type=network_type,
    )

    graph, blocked_edges = assign_flood_risk(
        graph,
        flood_raster,
        trust_score,
        block_depth_m=block_depth_m,
        penalty_strength=penalty_strength,
    )

    path = route(
        graph,
        start_lat,
        start_lon,
        end_lat,
        end_lon,
    )

    length = route_length(
        graph,
        path
    )

    geojson_path = (
        ROUTING_DIR
        / f"{output_name}.geojson"
    )

    save_route_geojson(
        graph,
        path,
        geojson_path
    )

    result = {
        "trust_score": float(
            trust_score
        ),
        "trust_mode": (
            "NORMAL"
            if trust_score >= 80
            else "CONSERVATIVE"
        ),
        "blocked_edges": int(
            blocked_edges
        ),
        "route_nodes": len(path),
        "route_length_m": float(
            length
        ),
        "start": {
            "latitude": start_lat,
            "longitude": start_lon,
        },
        "destination": {
            "latitude": end_lat,
            "longitude": end_lon,
        },
        "flood_block_threshold_m":
            block_depth_m,
        "route_file": str(
            geojson_path
        ),
    }

    json_path = (
        ROUTING_DIR
        / f"{output_name}.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    return result


if __name__ == "__main__":

    print(
        "Trust-aware evacuation router loaded successfully."
    )


