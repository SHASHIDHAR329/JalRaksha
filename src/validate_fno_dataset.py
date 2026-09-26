import numpy as np
from pathlib import Path

root = Path(r"C:\JalRaksha\outputs\fno_dataset")

X = np.load(root / "X.npy")
Y = np.load(root / "Y.npy")
M = np.load(root / "mask.npy")

print("=" * 70)
print("FNO DATASET VALIDATION")
print("=" * 70)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Mask shape:", M.shape)

print("X NaNs:", np.isnan(X).sum())
print("Y NaNs:", np.isnan(Y).sum())

print("X min/max:", float(X.min()), float(X.max()))
print("Y min/max:", float(Y.min()), float(Y.max()))

print("Valid mask cells:", int(M.sum()))
print("Total grid cells:", M.size)

print("\nScenario maximum depths:")
for i in range(Y.shape[0]):
    print(f"  {i+1:02d}: {Y[i,0].max():.6f} m")

print("\nMaximum-depth range:",
      float(Y[:,0].max(axis=(1,2)).min()),
      "to",
      float(Y[:,0].max(axis=(1,2)).max()),
      "m")

print("=" * 70)
