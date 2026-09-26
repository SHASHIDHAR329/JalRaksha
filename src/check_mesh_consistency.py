import pandas as pd
from pathlib import Path

root = Path(r"C:\JalRaksha\outputs\dflowfm")

names = ["jalraksha_flume"] + [
    f"jalraksha_flume_scenario_{i:02d}" for i in range(2, 11)
]

print("SCENARIO                    XMIN        XMAX        YMIN        YMAX     CELLS")
print("-" * 82)

for name in names:
    path = root / name / "flood_depth_cells.csv"
    d = pd.read_csv(path)

    print(
        f"{name:<28}"
        f"{d['x'].min():>10.3f}"
        f"{d['x'].max():>12.3f}"
        f"{d['y'].min():>12.3f}"
        f"{d['y'].max():>12.3f}"
        f"{len(d):>10}"
    )
