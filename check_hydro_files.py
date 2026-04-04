import os

base_path = "/scratch/user/lorenzo32/WATNEEDS+SMEW/WB_interpolated_first4hours"

scenarios = [
    ("vite",   "drip"),
    ("vite",   "surface"),
    ("vite",   "rainfed"),
    ("olivo",  "drip"),
    ("olivo",  "surface"),
    ("olivo",  "rainfed"),
    ("agrumi", "drip"),
    ("agrumi", "surface"),
    ("agrumi", "rainfed"),
    ("pesco",  "drip"),
    ("pesco",  "surface"),
    ("pesco",  "rainfed"),
    ("grano",  "rainfed"),
]

print(f"{'Scenario':<30} {'Path exists':<14} {'N files'}")
print("-" * 60)
for crop, irr in scenarios:
    folder = os.path.join(base_path, f"{crop}_{irr}")
    if os.path.exists(folder):
        n = len(os.listdir(folder))
        print(f"{crop}_{irr:<25} {'YES':<14} {n}")
    else:
        print(f"{crop}_{irr:<25} {'NO':<14} —")
