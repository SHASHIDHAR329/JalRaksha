from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(r"C:\JalRaksha")
EVAL_DIR = ROOT / r"outputs\fno_model\evaluation"
PLOT_DIR = EVAL_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

with open(EVAL_DIR / "validation_summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

scenarios = summary["validation_scenarios"]

for scenario in scenarios:

    truth_path = EVAL_DIR / f"{scenario}_truth.npy"
    pred_path = EVAL_DIR / f"{scenario}_prediction.npy"

    truth = np.load(truth_path)[0]
    pred = np.load(pred_path)[0]

    error = np.abs(pred - truth)

    vmax = max(float(truth.max()), float(pred.max()))

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_axes([0.03, 0.15, 0.28, 0.72])
    im1 = ax1.imshow(truth, origin="lower", aspect="auto", vmin=0, vmax=vmax)
    ax1.set_title(f"{scenario} - D-Flow FM Truth")
    ax1.set_xlabel("Grid X")
    ax1.set_ylabel("Grid Y")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Depth (m)")

    ax2 = fig.add_axes([0.36, 0.15, 0.28, 0.72])
    im2 = ax2.imshow(pred, origin="lower", aspect="auto", vmin=0, vmax=vmax)
    ax2.set_title(f"{scenario} - FNO Prediction")
    ax2.set_xlabel("Grid X")
    ax2.set_ylabel("Grid Y")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Depth (m)")

    ax3 = fig.add_axes([0.69, 0.15, 0.28, 0.72])
    im3 = ax3.imshow(error, origin="lower", aspect="auto")
    ax3.set_title(f"{scenario} - Absolute Error")
    ax3.set_xlabel("Grid X")
    ax3.set_ylabel("Grid Y")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Error (m)")

    fig.suptitle("JalRaksha FNO Validation", fontsize=14)

    output = PLOT_DIR / f"{scenario}_validation.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)

print("=" * 70)
print("FNO VISUAL VALIDATION COMPLETE")
print("Output:", PLOT_DIR)
print("=" * 70)
