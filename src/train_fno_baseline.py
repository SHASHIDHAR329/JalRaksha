from pathlib import Path
import json
import csv

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from neuralop.models import FNO


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\JalRaksha")
DATA_DIR = ROOT / r"outputs\fno_dataset"
MODEL_DIR = ROOT / r"outputs\fno_model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("JALRAKSHA FNO BASELINE TRAINING")
print("=" * 70)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(DATA_DIR / "X.npy").astype(np.float32)
Y = np.load(DATA_DIR / "Y.npy").astype(np.float32)
MASK = np.load(DATA_DIR / "mask.npy").astype(np.float32)

with open(DATA_DIR / "scenario_ids.json", "r", encoding="utf-8") as f:
    scenario_ids = json.load(f)

print("\nDataset:")
print("X:", X.shape)
print("Y:", Y.shape)
print("Mask:", MASK.shape)


# ============================================================
# CHECK DATA
# ============================================================

if X.ndim != 4:
    raise ValueError(f"Expected X to have 4 dimensions, got {X.shape}")

if Y.ndim != 4:
    raise ValueError(f"Expected Y to have 4 dimensions, got {Y.shape}")

if X.shape[0] != Y.shape[0]:
    raise ValueError("X and Y scenario counts do not match.")

if np.isnan(X).any():
    raise ValueError("X contains NaN values.")

if np.isnan(Y).any():
    raise ValueError("Y contains NaN values.")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

n_samples = X.shape[0]

if n_samples != 10:
    print(f"WARNING: Expected 10 scenarios, found {n_samples}.")

rng = np.random.default_rng(SEED)
indices = rng.permutation(n_samples)

n_train = int(round(0.8 * n_samples))

train_idx = indices[:n_train]
val_idx = indices[n_train:]

print("\nTrain scenarios:")
for i in train_idx:
    print(" ", scenario_ids[i])

print("\nValidation scenarios:")
for i in val_idx:
    print(" ", scenario_ids[i])


# ============================================================
# TARGET SCALING
# ============================================================

target_scale = float(Y[train_idx].max())

if target_scale <= 0:
    raise ValueError("Training target maximum is not positive.")

Y_scaled = Y / target_scale


# ============================================================
# TORCH DATA
# ============================================================

X_train = torch.from_numpy(X[train_idx])
Y_train = torch.from_numpy(Y_scaled[train_idx])

X_val = torch.from_numpy(X[val_idx])
Y_val = torch.from_numpy(Y_scaled[val_idx])

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = FNO(
    n_modes=(8, 16),
    hidden_channels=32,
    in_channels=7,
    out_channels=1,
    n_layers=4,
).to(DEVICE)

print("\nModel created:")
print(model)


# ============================================================
# OPTIMIZER
# ============================================================

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


# ============================================================
# MASK
# ============================================================

mask_tensor = torch.from_numpy(MASK).to(DEVICE)
mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)


# ============================================================
# LOSS
# ============================================================

def masked_mse(pred, target):
    error = (pred - target) ** 2
    numerator = (error * mask_tensor).sum()
    denominator = mask_tensor.sum() * pred.shape[0]

    return numerator / denominator


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 500
PATIENCE = 80

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

history = []

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):

    model.train()

    train_loss_sum = 0.0
    train_count = 0

    for xb, yb in train_loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        pred = model(xb)

        loss = masked_mse(pred, yb)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = xb.shape[0]
        train_loss_sum += loss.item() * batch_size
        train_count += batch_size

    train_loss = train_loss_sum / train_count

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)

            loss = masked_mse(pred, yb)

            batch_size = xb.shape[0]
            val_loss_sum += loss.item() * batch_size
            val_count += batch_size

    val_loss = val_loss_sum / val_count

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
        epochs_without_improvement = 0

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
            },
            MODEL_DIR / "fno_baseline_best.pt"
        )

    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Train {train_loss:.8f} | "
            f"Val {val_loss:.8f} | "
            f"LR {lr:.2e}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch}. "
            f"Best epoch: {best_epoch}"
        )
        break


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = MODEL_DIR / "training_history.csv"

with history_path.open("w", encoding="utf-8", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate",
        ]
    )

    writer.writeheader()
    writer.writerows(history)


# ============================================================
# SAVE SPLIT
# ============================================================

split_info = {
    "seed": SEED,
    "train_indices": train_idx.tolist(),
    "validation_indices": val_idx.tolist(),
    "train_scenarios": [scenario_ids[i] for i in train_idx],
    "validation_scenarios": [scenario_ids[i] for i in val_idx],
    "target_scale": target_scale,
}

with (MODEL_DIR / "train_val_split.json").open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(split_info, f, indent=2)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FNO TRAINING COMPLETE")
print("=" * 70)
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
print("Model:", MODEL_DIR / "fno_baseline_best.pt")
print("History:", history_path)
print("Split:", MODEL_DIR / "train_val_split.json")
print("Target scale:", target_scale)
print("=" * 70)
