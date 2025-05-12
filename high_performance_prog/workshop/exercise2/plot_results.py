import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("fig", exist_ok=True)

df = pd.read_csv("cold_times.csv")
repeats = df["repeats"].unique()
for r in repeats:
    fig, ax = plt.subplots()
    for m in df["method"].unique():
        sub = df[df["method"] == m]
        sub = sub[sub["repeats"] == r]
        sub["h"] = sub["resolution"].map(lambda x: x.split("x")[0])
        ax.plot(sub["h"], sub["time_s"], marker='o', label=m)
    ax.set_title(f"Performance for {r} image(s)")
    ax.set_xlabel("Width and Height (pixels)")
    ax.set_ylabel("Time (s)")
    ax.legend()
    plt.show()
    plt.savefig(os.path.join("fig", f"comp_{r}.png"))
