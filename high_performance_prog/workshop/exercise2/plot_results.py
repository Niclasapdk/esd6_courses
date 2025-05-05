import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cold_times.csv")
fig, ax = plt.subplots()
for m in df["method"].unique():
    sub = df[df["method"] == m]
    ax.plot(sub["grid"], sub["time_s"], marker='o', label=m)
ax.set_xlabel("Grid size")
ax.set_ylabel("Time (s)")
ax.legend()
plt.show()
