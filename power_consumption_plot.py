import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load the data
data = pd.DataFrame({
    "load": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "vWall": [25, 35, 50, 60, 70, 80, 90, 100, 115, 140],
    "Raspberry Pi3": [4.0, 4.1, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.3, 5.4],
    "Raspberry Pi4": [5.5, 5.8, 6.0, 6.5, 6.7, 6.9, 7.1, 7.3, 7.5, 7.7],
    "Shuttle": [9, 13, 11, 10, 18, 21, 23, 23, 22, 22],
    "Intel Nuc": [7, 11, 10, 9, 10, 12, 30, 35, 37, 38],
})

# Reshape data for seaborn
df_melted = data.melt(id_vars="load", var_name="Cluster Type", value_name="Power Consumption")

# Plotting with seaborn
plt.figure(figsize=(7, 5))
sns.lineplot(
    data=df_melted,
    x="load",
    y="Power Consumption",
    hue="Cluster Type",
    style="Cluster Type",
    markers=True,
    dashes=False,
    palette="tab10"
)

# Plotting
# plt.figure(figsize=(10, 6))
# for col in data.columns[1:]:
#     plt.plot(data["load"], data[col], marker='o', label=col)

# plt.title(r'\textbf{Power Consumption vs CPU Load}', fontsize=20)
plt.xlabel(r'\textbf{CPU Load [\%]}', fontsize=20)
plt.ylabel(r'\textbf{Power Consumption [Watts]}', fontsize=20)
plt.legend(title="Cluster Types", fontsize=12, title_fontsize=14, ncols=2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
#plt.show()
plt.savefig("power_consumption_plot.pdf", bbox_inches="tight", dpi=250)

