import pandas as pd
import matplotlib.pyplot as plt

readings = pd.read_parquet("data/raw/readings.parquet")
metadata = pd.read_parquet("data/raw/batch_metadata.parquet")

batch_id = readings["batch_id"].unique()[0]
batch = readings[readings["batch_id"] == batch_id].sort_values("timestamp")

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(batch["timestamp"], batch["temperature"])
axes[0].set_ylabel("Temperature (°C)")
axes[0].set_title(f"Batch {batch_id}")

axes[1].plot(batch["timestamp"], batch["ppfd"])
axes[1].set_ylabel("PPFD (µmol/m²/s)")
axes[1].set_xlabel("Timestamp")

plt.tight_layout()
plt.savefig("batch_plot.png")
plt.show()