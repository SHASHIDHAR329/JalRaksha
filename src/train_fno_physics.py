from pathlib import Path
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from neuralop.models import FNO


ROOT = Path(r"C:\JalRaksha")
DATA_DIR = ROOT / r"outputs\fno_dataset"
MASKED_DIR = ROOT / r"outputs\fno_model_masked"
MODEL_DIR = ROOT / r"outputs\fno_model_physics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("JALRAKSHA PHYSICS-AWARE FNO TRAINING")
print("=" * 70)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

X = np.load(DATA_DIR / "X.npy").astype(np.float32)
Y = np.load(DATA_DIR / "Y.npy").astype(np.float32)
MASK = np.load(DATA_DIR / "mask.npy").astype(np.float32)

with open(DATA_DIR / "scenario_ids.json", "r", encoding="utf-8") as f:
    scenario_ids = json.load(f)

print("X:", X.shape)
print("Y:", Y.shape)
print("Mask:", MASK.shape)


# ------------------------------------------------------------
# Fixed split
# ------------------------------------------------------------

indices = np.random.default_rng(SEED).permutation(len(X))
n_train = int(round(0.8 * len(X)))

train_idx = indices[:n_train]
val_idx = indices[n_train:]

print("\nTraining scenarios:")
for i in train_idx:
    print(" ", scenario_ids[i])

print("\nValidation scenarios:")
for i in val_idx:
    print(" ", scenario_ids[i])


# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------

target_scale = float(Y[train_idx].max())

Y_scaled = Y / target_scale

X_train = torch.from_numpy(X[train_idx])
Y_train = torch.from_numpy(Y_scaled[train_idx])

X_val = torch.from_numpy(X[val_idx])
Y_val = torch.from_numpy(Y_scaled[val_idx])

train_loader = DataLoader(
    TensorDataset(X_train, Y_train),
    batch_size=2,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, Y_val),
    batch_size=2,
    shuffle=False
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = FNO(
    n_modes=(8, 16),
    hidden_channels=32,
    in_channels=7,
    out_channels=1,
    n_layers=4,
).to(DEVICE)


# ------------------------------------------------------------
# Start from successful mask-constrained model
# ------------------------------------------------------------

checkpoint_path = MASKED_DIR / "fno_masked_best.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

print("\nInitialized from:")
print(checkpoint_path)


# ------------------------------------------------------------
# Physical mask
# ------------------------------------------------------------

mask = torch.from_numpy(MASK).to(DEVICE)
mask = mask.unsqueeze(0).unsqueeze(0)


# ------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------

def masked_mse(pred, target):
    error = (pred - target) ** 2
    return (error * mask).sum() / (mask.sum() * pred.shape[0])


def gradient_loss(pred, target):
    pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]

    targ_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    targ_x = target[:, :, :, 1:] - target[:, :, :, :-1]

    mask_y = mask[:, :, 1:, :]
    mask_x = mask[:, :, :, 1:]

    loss_y = (
        ((pred_y - targ_y) ** 2) * mask_y
    ).sum() / (mask_y.sum() * pred.shape[0])

    loss_x = (
        ((pred_x - targ_x) ** 2) * mask_x
    ).sum() / (mask_x.sum() * pred.shape[0])

    return 0.5 * (loss_x + loss_y)


def integrated_depth_loss(pred, target):
    pred_mean = (pred * mask).sum(dim=(1, 2, 3)) / mask.sum()
    target_mean = (target * mask).sum(dim=(1, 2, 3)) / mask.sum()

    return F.mse_loss(pred_mean, target_mean)


# Physics-aware weighting
LAMBDA_GRAD = 0.05
LAMBDA_INTEGRAL = 0.02


def total_loss(pred, target):
    data = masked_mse(pred, target)
    grad = gradient_loss(pred, target)
    integral = integrated_depth_loss(pred, target)

    total = (
        data
        + LAMBDA_GRAD * grad
        + LAMBDA_INTEGRAL * integral
    )

    return total, data, grad, integral


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=25
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

EPOCHS = 300
PATIENCE = 60

best_val = float("inf")
best_epoch = 0
no_improvement = 0

history = []

for epoch in range(1, EPOCHS + 1):

    model.train()

    train_total = 0.0
    train_data = 0.0
    train_grad = 0.0
    train_integral = 0.0
    train_count = 0

    for xb, yb in train_loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        raw = model(xb)

        # Enforce physical domain and non-negative depth
        pred = torch.relu(raw) * mask

        loss, data_loss, grad_loss, integral_loss = total_loss(
            pred,
            yb
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        bs = xb.shape[0]

        train_total += loss.item() * bs
        train_data += data_loss.item() * bs
        train_grad += grad_loss.item() * bs
        train_integral += integral_loss.item() * bs
        train_count += bs

    train_total /= train_count
    train_data /= train_count
    train_grad /= train_count
    train_integral /= train_count


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_total = 0.0
    val_data = 0.0
    val_grad = 0.0
    val_integral = 0.0
    val_count = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            raw = model(xb)
            pred = torch.relu(raw) * mask

            loss, data_loss, grad_loss, integral_loss = total_loss(
                pred,
                yb
            )

            bs = xb.shape[0]

            val_total += loss.item() * bs
            val_data += data_loss.item() * bs
            val_grad += grad_loss.item() * bs
            val_integral += integral_loss.item() * bs
            val_count += bs

    val_total /= val_count
    val_data /= val_count
    val_grad /= val_count
    val_integral /= val_count

    scheduler.step(val_total)

    lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_total": train_total,
        "train_data": train_data,
        "train_gradient": train_grad,
        "train_integral": train_integral,
        "val_total": val_total,
        "val_data": val_data,
        "val_gradient": val_grad,
        "val_integral": val_integral,
        "learning_rate": lr,
    })

    if val_total < best_val:

        best_val = val_total
        best_epoch = epoch
        no_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "target_scale": target_scale,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "scenario_ids": scenario_ids,
                "input_shape": list(X.shape),
                "output_shape": list(Y.shape),
                "mask_shape": list(MASK.shape),
                "n_modes": (8, 16),
                "hidden_channels": 32,
                "in_channels": 7,
                "out_channels": 1,
                "n_layers": 4,
                "output_mask_constraint": True,
                "nonnegative_constraint": True,
                "lambda_gradient": LAMBDA_GRAD,
                "lambda_integral": LAMBDA_INTEGRAL,
            },
            MODEL_DIR / "fno_physics_best.pt"
        )

    else:
        no_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Train={train_total:.8f} | "
            f"Val={val_total:.8f} | "
            f"Data={val_data:.8f} | "
            f"Grad={val_grad:.8f} | "
            f"Integral={val_integral:.8f} | "
            f"LR={lr:.2e}"
        )

    if no_improvement >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch}. "
            f"Best epoch: {best_epoch}"
        )
        break


# ------------------------------------------------------------
# Save history
# ------------------------------------------------------------

history_path = MODEL_DIR / "training_history.csv"

with history_path.open("w", encoding="utf-8", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(history[0].keys())
    )

    writer.writeheader()
    writer.writerows(history)


# ------------------------------------------------------------
# Save split information
# ------------------------------------------------------------

split_info = {
    "seed": SEED,
    "train_indices": train_idx.tolist(),
    "validation_indices": val_idx.tolist(),
    "train_scenarios": [scenario_ids[i] for i in train_idx],
    "validation_scenarios": [scenario_ids[i] for i in val_idx],
    "target_scale": target_scale,
    "lambda_gradient": LAMBDA_GRAD,
    "lambda_integral": LAMBDA_INTEGRAL,
}

with (MODEL_DIR / "train_val_split.json").open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(split_info, f, indent=2)


print("\n" + "=" * 70)
print("PHYSICS-AWARE FNO TRAINING COMPLETE")
print("=" * 70)
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val)
print("Model:", MODEL_DIR / "fno_physics_best.pt")
print("History:", history_path)
print("=" * 70)
