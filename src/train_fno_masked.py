from pathlib import Path
import json
import csv

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from neuralop.models import FNO


ROOT = Path(r"C:\JalRaksha")
DATA_DIR = ROOT / r"outputs\fno_dataset"
MODEL_DIR = ROOT / r"outputs\fno_model_masked"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("JALRAKSHA MASK-CONSTRAINED FNO TRAINING")
print("=" * 70)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ------------------------------------------------------------
# Load data
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
# Fixed train/validation split
# ------------------------------------------------------------

indices = np.random.default_rng(SEED).permutation(len(X))

n_train = int(round(0.8 * len(X)))

train_idx = indices[:n_train]
val_idx = indices[n_train:]

print("\nTraining:")
for i in train_idx:
    print(" ", scenario_ids[i])

print("\nValidation:")
for i in val_idx:
    print(" ", scenario_ids[i])


# ------------------------------------------------------------
# Target scaling
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
# Physical output mask
# ------------------------------------------------------------

mask_tensor = (
    torch.from_numpy(MASK)
    .float()
    .to(DEVICE)
    .unsqueeze(0)
    .unsqueeze(0)
)

print("\nMask-constrained output enabled.")


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=25
)


def masked_mse(pred, target):
    error = (pred - target) ** 2
    numerator = (error * mask_tensor).sum()
    denominator = mask_tensor.sum() * pred.shape[0]
    return numerator / denominator


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

EPOCHS = 500
PATIENCE = 80

best_val_loss = float("inf")
best_epoch = 0
no_improvement = 0

history = []

for epoch in range(1, EPOCHS + 1):

    model.train()

    train_total = 0.0
    train_count = 0

    for xb, yb in train_loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        raw_pred = model(xb)

        # Enforce physical domain constraint
        pred = raw_pred * mask_tensor

        loss = masked_mse(pred, yb)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        bs = xb.shape[0]
        train_total += loss.item() * bs
        train_count += bs

    train_loss = train_total / train_count


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_total = 0.0
    val_count = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            raw_pred = model(xb)

            # Same physical constraint during validation
            pred = raw_pred * mask_tensor

            loss = masked_mse(pred, yb)

            bs = xb.shape[0]
            val_total += loss.item() * bs
            val_count += bs

    val_loss = val_total / val_count

    scheduler.step(val_loss)

    lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": lr,
    })

    if val_loss < best_val_loss:

        best_val_loss = val_loss
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
            },
            MODEL_DIR / "fno_masked_best.pt"
        )

    else:
        no_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Train {train_loss:.8f} | "
            f"Val {val_loss:.8f} | "
            f"LR {lr:.2e}"
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

with (MODEL_DIR / "training_history.csv").open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate"
        ]
    )

    writer.writeheader()
    writer.writerows(history)


# ------------------------------------------------------------
# Save split
# ------------------------------------------------------------

split_info = {
    "seed": SEED,
    "train_indices": train_idx.tolist(),
    "validation_indices": val_idx.tolist(),
    "train_scenarios": [scenario_ids[i] for i in train_idx],
    "validation_scenarios": [scenario_ids[i] for i in val_idx],
    "target_scale": target_scale,
    "output_mask_constraint": True,
}

with (MODEL_DIR / "train_val_split.json").open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(split_info, f, indent=2)


print("\n" + "=" * 70)
print("MASK-CONSTRAINED FNO TRAINING COMPLETE")
print("=" * 70)
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
print("Model:", MODEL_DIR / "fno_masked_best.pt")
print("History:", MODEL_DIR / "training_history.csv")
print("=" * 70)
