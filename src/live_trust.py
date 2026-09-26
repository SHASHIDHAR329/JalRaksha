from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(r"C:\JalRaksha")

DATA_DIR = ROOT / r"outputs\fno_dataset"
MANIFEST_FILE = ROOT / r"outputs\dataset_manifest.csv"


def load_trust_context():

    mask = np.load(
        DATA_DIR / "mask.npy"
    ).astype(bool)

    x_grid = np.load(
        DATA_DIR / "x_grid.npy"
    ).astype(np.float32)

    y_grid = np.load(
        DATA_DIR / "y_grid.npy"
    ).astype(np.float32)

    manifest = pd.read_csv(MANIFEST_FILE)

    dx = float(np.mean(np.diff(x_grid)))
    dy = float(np.mean(np.diff(y_grid)))
    cell_area = abs(dx * dy)

    train_masked_depths = []

    # Load the training scenarios from the existing split
    split_file = (
        ROOT
        / r"outputs\fno_model_physics\train_val_split.json"
    )

    with split_file.open(
        "r",
        encoding="utf-8"
    ) as f:
        split = json.load(f)

    train_indices = split["train_indices"]

    # The FNO training targets
    Y = np.load(
        DATA_DIR / "Y.npy"
    ).astype(np.float32)

    scenario_ids = json.load(
        open(
            DATA_DIR / "scenario_ids.json",
            "r",
            encoding="utf-8"
        )
    )

    for idx in train_indices:
        train_masked_depths.append(
            float(Y[idx, 0][mask].max())
        )

    train_min_depth = float(
        min(train_masked_depths)
    )

    train_max_depth = float(
        max(train_masked_depths)
    )

    # Training spatial-gradient envelope
    gradient_means = []

    for idx in train_indices:

        depth = Y[idx, 0]

        gy, gx = np.gradient(depth)

        gradient = np.sqrt(
            gx ** 2 + gy ** 2
        )

        gradient_means.append(
            float(
                gradient[mask].mean()
            )
        )

    gradient_low = float(
        np.percentile(
            gradient_means,
            5
        )
    )

    gradient_high = float(
        np.percentile(
            gradient_means,
            95
        )
    )

    # Hydrograph/depth consistency proxy
    manifest_index = manifest.set_index(
        "scenario"
    )

    hydro_volumes = []

    depth_proxies = []

    for idx in train_indices:

        scenario = scenario_ids[idx]

        hydro_volume = float(
            manifest_index.loc[
                scenario,
                "hydrograph_volume_m3"
            ]
        )

        depth = Y[idx, 0]

        proxy = float(
            depth[mask].sum()
            * cell_area
        )

        hydro_volumes.append(
            hydro_volume
        )

        depth_proxies.append(
            proxy
        )

    A = np.column_stack(
        [
            hydro_volumes,
            np.ones(len(hydro_volumes))
        ]
    )

    coef, _, _, _ = np.linalg.lstsq(
        A,
        depth_proxies,
        rcond=None
    )

    return {
        "mask": mask,
        "cell_area": cell_area,
        "train_min_depth": train_min_depth,
        "train_max_depth": train_max_depth,
        "gradient_low": gradient_low,
        "gradient_high": gradient_high,
        "proxy_slope": float(coef[0]),
        "proxy_intercept": float(coef[1]),
    }


def evaluate_trust(
    prediction,
    peak_q_m3s,
    hydrograph_volume_m3,
):

    context = load_trust_context()

    mask = context["mask"]

    valid_prediction = prediction[mask]

    # --------------------------------------------------------
    # CHECK 1: non-negative depth
    # --------------------------------------------------------

    min_depth = float(
        valid_prediction.min()
    )

    if min_depth >= -0.001:
        nonnegative_score = 20.0
        nonnegative_status = "PASS"
    elif min_depth >= -0.01:
        nonnegative_score = 10.0
        nonnegative_status = "REVIEW"
    else:
        nonnegative_score = 0.0
        nonnegative_status = "FAIL"

    # --------------------------------------------------------
    # CHECK 2: domain mask
    # --------------------------------------------------------

    outside = prediction[~mask]

    outside_max = (
        float(np.max(np.abs(outside)))
        if len(outside)
        else 0.0
    )

    if outside_max <= 0.001:
        mask_score = 20.0
        mask_status = "PASS"
    elif outside_max <= 0.005:
        mask_score = 10.0
        mask_status = "REVIEW"
    else:
        mask_score = 0.0
        mask_status = "FAIL"

    # --------------------------------------------------------
    # CHECK 3: depth envelope
    # --------------------------------------------------------

    predicted_max = float(
        valid_prediction.max()
    )

    lower = (
        context["train_min_depth"]
        * 0.75
    )

    upper = (
        context["train_max_depth"]
        * 1.25
    )

    if lower <= predicted_max <= upper:

        range_score = 20.0
        range_status = "PASS"

    elif (
        context["train_min_depth"] * 0.50
        <= predicted_max
        <= context["train_max_depth"] * 1.50
    ):

        range_score = 10.0
        range_status = "REVIEW"

    else:

        range_score = 0.0
        range_status = "FAIL"

    # --------------------------------------------------------
    # CHECK 4: hydrograph/depth proxy
    # --------------------------------------------------------

    predicted_proxy = float(
        valid_prediction.sum()
        * context["cell_area"]
    )

    expected_proxy = (
        context["proxy_slope"]
        * hydrograph_volume_m3
        + context["proxy_intercept"]
    )

    if expected_proxy > 0:

        proxy_ratio = (
            predicted_proxy
            / expected_proxy
        )

    else:

        proxy_ratio = float("inf")

    if 0.70 <= proxy_ratio <= 1.30:

        proxy_score = 20.0
        proxy_status = "PASS"

    elif 0.50 <= proxy_ratio <= 1.50:

        proxy_score = 10.0
        proxy_status = "REVIEW"

    else:

        proxy_score = 0.0
        proxy_status = "FAIL"

    # --------------------------------------------------------
    # CHECK 5: spatial smoothness
    # --------------------------------------------------------

    gy, gx = np.gradient(
        prediction
    )

    gradient = np.sqrt(
        gx ** 2 + gy ** 2
    )

    gradient_mean = float(
        gradient[mask].mean()
    )

    if (
        context["gradient_low"]
        <= gradient_mean
        <= context["gradient_high"]
    ):

        smoothness_score = 20.0
        smoothness_status = "PASS"

    elif (
        context["gradient_low"] * 0.5
        <= gradient_mean
        <= context["gradient_high"] * 2.0
    ):

        smoothness_score = 10.0
        smoothness_status = "REVIEW"

    else:

        smoothness_score = 0.0
        smoothness_status = "FAIL"

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    score = (
        nonnegative_score
        + mask_score
        + range_score
        + proxy_score
        + smoothness_score
    )

    if score >= 80:
        status = "TRUSTED"
    elif score >= 60:
        status = "REVIEW"
    else:
        status = "REJECT"

    return {
        "score": float(score),
        "status": status,

        "checks": {
            "nonnegative": {
                "score": nonnegative_score,
                "status": nonnegative_status,
            },
            "domain_mask": {
                "score": mask_score,
                "status": mask_status,
            },
            "depth_range": {
                "score": range_score,
                "status": range_status,
            },
            "hydrograph_consistency": {
                "score": proxy_score,
                "status": proxy_status,
            },
            "spatial_smoothness": {
                "score": smoothness_score,
                "status": smoothness_status,
            },
        },

        "diagnostics": {
            "minimum_depth_m": min_depth,
            "outside_mask_max_depth_m": outside_max,
            "predicted_max_depth_m": predicted_max,
            "predicted_depth_proxy_m3": predicted_proxy,
            "expected_depth_proxy_m3": expected_proxy,
            "proxy_ratio": float(proxy_ratio),
            "gradient_mean": gradient_mean,
        },
    }


if __name__ == "__main__":

    print(
        "Live Trust Engine module loaded successfully."
    )
