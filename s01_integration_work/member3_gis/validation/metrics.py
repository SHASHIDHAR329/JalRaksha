import numpy as np


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def csi(pred, true, threshold=0.1):
    pred_flood = pred >= threshold
    true_flood = true >= threshold

    intersection = np.logical_and(pred_flood, true_flood).sum()
    union = np.logical_or(pred_flood, true_flood).sum()

    return float(intersection / union) if union else 1.0


def pod(pred, true, threshold=0.1):
    pred_flood = pred >= threshold
    true_flood = true >= threshold

    hits = np.logical_and(pred_flood, true_flood).sum()
    misses = np.logical_and(~pred_flood, true_flood).sum()

    return float(hits / (hits + misses)) if (hits + misses) else 1.0


def far(pred, true, threshold=0.1):
    pred_flood = pred >= threshold
    true_flood = true >= threshold

    false_alarms = np.logical_and(pred_flood, ~true_flood).sum()
    hits = np.logical_and(pred_flood, true_flood).sum()

    return float(
        false_alarms / (hits + false_alarms)
    ) if (hits + false_alarms) else 0.0


def iou(pred, true, threshold=0.1):
    pred_flood = pred >= threshold
    true_flood = true >= threshold

    intersection = np.logical_and(pred_flood, true_flood).sum()
    union = np.logical_or(pred_flood, true_flood).sum()

    return float(intersection / union) if union else 1.0