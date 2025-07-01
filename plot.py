import os
from collections import defaultdict

import pandas as pd
import seaborn as sns
from pandas import CategoricalDtype
import matplotlib
from matplotlib import pyplot as plt
import re

matplotlib.use('TkAgg')
import numpy as np
import scipy.stats as stats

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# Calculate the 95% confidence interval
def calculate_95_ci(series):
    mean = series.mean()
    std = series.std()
    n = len(series)
    if n > 1:
        ci = stats.t.ppf(0.975, n - 1) * (std / (n ** 0.5))  # 95% CI
    else:
        ci = 0  # If there's only one sample, CI cannot be computed
    return mean, std, ci


def process_file(filepath, test_name):
    data = pd.read_csv(filepath)

    # Assign the file-based test name
    data['Test'] = test_name

    metrics = {
        'reward': 'reward',
        # 'ep_block_prob': 'ep_block_prob',
        'ep_accepted_requests': 'ep_accepted_requests',
        'ep_rejected_requests': 'ep_rejected_requests',
        # 'ep_deploy_all': 'ep_deploy_all',
        # 'ep_fdd': 'ep_fdd',
        'avg_latency': 'avg_latency',
        'avg_cost': 'avg_cost',
        'avg_cpu_cluster_selected': 'avg_cpu_cluster_selected',
        'gini': 'gini'
    }

    print(f"\n===== {test_name} =====")
    for metric, description in metrics.items():
        if metric not in data.columns:
            continue  # Skip missing metrics

        series = data[metric]
        mean, std, ci = calculate_95_ci(series)
        print(f"{description}: {mean:.2f} $\pm$ {ci:.2f}")


def main():
    dir = 'results/'
    for filename in os.listdir(dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(dir, filename)
            test_name = os.path.splitext(filename)[0]  # Remove .csv
            process_file(filepath, test_name)

    # Data
    data = [
        # Format: (agent_config, metric_name, mean, std)
        ("dqn_c8_r4", 8, 4, "DQN (DS)", "reward", 54.52, 4.81),
        ("dqn_c8_r4", 8, 4, "DQN (DS)","ep_block_prob", 0.02, 0.03),
        ("dqn_c8_r4", 8, 4, "DQN (DS)","ep_accepted_requests", 97.60, 2.74),
        ("dqn_c8_r4", 8, 4, "DQN (DS)","ep_rejected_requests", 2.40, 2.74),
        ("dqn_c8_r4", 8, 4, "DQN (DS)","ep_deploy_all", 91.00, 7.32),
        ("dqn_c8_r4", 8, 4, "DQN (DS)","ep_fdd", 6.60, 4.69),
        ("dqn_c8_r4", 8, 4, "DQN (DS)","avg_latency", 456.42, 12.15),
        ("dqn_c8_r4", 8, 4, "DQN (DS)","avg_cost", 10.66, 2.44),
        ("dqn_c8_r4", 8, 4, "DQN (DS)","avg_cpu_cluster_selected", 13.39, 0.62),
        ("dqn_c8_r4", 8, 4, "DQN (DS)", "gini", 0.65, 0.04),

        ("dqn_c8_r6", 8, 6, "DQN (DS)", "reward", 55.84, 3.25),
        ("dqn_c8_r6", 8, 6, "DQN (DS)", "ep_block_prob", 0.02, 0.02),
        ("dqn_c8_r6", 8, 6, "DQN (DS)", "ep_accepted_requests", 98.37, 1.64),
        ("dqn_c8_r6", 8, 6, "DQN (DS)", "ep_rejected_requests", 1.63, 1.64),
        ("dqn_c8_r6", 8, 6, "DQN (DS)", "ep_deploy_all", 89.57, 7.52),
        ("dqn_c8_r6", 8, 6, "DQN (DS)", "ep_fdd", 8.80, 5.94),
        ("dqn_c8_r6", 8, 6, "DQN (DS)", "avg_latency", 448.02, 11.91),
        ("dqn_c8_r6", 8, 6, "DQN (DS)", "avg_cost", 11.22, 2.43),
        ("dqn_c8_r6", 8, 6, "DQN (DS)", "avg_cpu_cluster_selected", 17.82, 1.07),
        ("dqn_c8_r6", 8, 6, "DQN (DS)", "gini", 0.65, 0.03),

        ("dqn_c8_r8", 8, 8, "DQN (DS)", "reward", 55.78, 2.52),
        ("dqn_c8_r8", 8, 8, "DQN (DS)", "ep_block_prob", 0.01, 0.01),
        ("dqn_c8_r8", 8, 8, "DQN (DS)", "ep_accepted_requests", 98.53, 1.13),
        ("dqn_c8_r8", 8, 8, "DQN (DS)", "ep_rejected_requests", 1.47, 1.13),
        ("dqn_c8_r8", 8, 8, "DQN (DS)", "ep_deploy_all", 85.57, 8.04),
        ("dqn_c8_r8", 8, 8, "DQN (DS)", "ep_fdd", 12.97, 6.98),
        ("dqn_c8_r8", 8, 8, "DQN (DS)", "avg_latency", 442.93, 15.03),
        ("dqn_c8_r8", 8, 8, "DQN (DS)", "avg_cost", 12.07, 2.23),
        ("dqn_c8_r8", 8, 8, "DQN (DS)", "avg_cpu_cluster_selected", 21.18, 1.40),
        ("dqn_c8_r8", 8, 8, "DQN (DS)", "gini", 0.66, 0.03),

        # --- Geometric entries ---
        ("geo_c8_r4", 8, 4, "Geometric (MORL)", "ep_accepted_requests", 242.30, 7.74),
        ("geo_c8_r4", 8, 4, "Geometric (MORL)", "ep_rejected_requests", 7.70, 7.74),
        ("geo_c8_r4", 8, 4, "Geometric (MORL)", "avg_latency", 576.33, 65.50),
        ("geo_c8_r4", 8, 4, "Geometric (MORL)", "avg_cost", 38.30, 5.48),
        ("geo_c8_r4", 8, 4, "Geometric (MORL)", "avg_cpu_cluster_selected", 19.08, 1.44),
        ("geo_c8_r4", 8, 4, "Geometric (MORL)", "gini", 0.52, 0.11),

        ("geo_c8_r6", 8, 6, "Geometric (MORL)", "ep_accepted_requests", 222.10, 23.33),
        ("geo_c8_r6", 8, 6, "Geometric (MORL)", "ep_rejected_requests", 27.90, 23.33),
        ("geo_c8_r6", 8, 6, "Geometric (MORL)", "avg_latency", 549.75, 65.66),
        ("geo_c8_r6", 8, 6, "Geometric (MORL)", "avg_cost", 41.18, 3.49),
        ("geo_c8_r6", 8, 6, "Geometric (MORL)", "avg_cpu_cluster_selected", 22.85, 2.77),
        ("geo_c8_r6", 8, 6, "Geometric (MORL)", "gini", 0.53, 0.11),

        ("geo_c8_r8", 8, 8, "Geometric (MORL)", "ep_accepted_requests", 228.90, 5.46),
        ("geo_c8_r8", 8, 8, "Geometric (MORL)", "ep_rejected_requests", 21.10, 5.46),
        ("geo_c8_r8", 8, 8, "Geometric (MORL)", "avg_latency", 656.93, 43.64),
        ("geo_c8_r8", 8, 8, "Geometric (MORL)", "avg_cost", 22.70, 5.90),
        ("geo_c8_r8", 8, 8, "Geometric (MORL)", "avg_cpu_cluster_selected", 33.23, 4.61),
        ("geo_c8_r8", 8, 8, "Geometric (MORL)", "gini", 0.51, 0.09),

        ("ppo_c8_r4", 8, 4, "PPO (DS)", "reward", 39.70, 0.61),
        ("ppo_c8_r4", 8, 4, "PPO (DS)", "ep_block_prob", 0.00, 0.00),
        ("ppo_c8_r4", 8, 4, "PPO (DS)", "ep_accepted_requests", 100.00, 0.00),
        ("ppo_c8_r4", 8, 4, "PPO (DS)", "ep_rejected_requests", 0.00, 0.00),
        ("ppo_c8_r4", 8, 4, "PPO (DS)", "ep_deploy_all", 100.00, 0.00),
        ("ppo_c8_r4", 8, 4, "PPO (DS)", "ep_fdd", 0.00, 0.00),
        ("ppo_c8_r4", 8, 4, "PPO (DS)", "avg_latency", 595.17, 3.19),
        ("ppo_c8_r4", 8, 4, "PPO (DS)", "avg_cost", 7.29, 0.27),
        ("ppo_c8_r4", 8, 4, "PPO (DS)", "avg_cpu_cluster_selected", 24.73, 0.10),
        ("ppo_c8_r4", 8, 4, "PPO (DS)", "gini", 0.79, 0.00),

        ("ppo_c8_r6", 8, 6, "PPO (DS)", "reward", 58.31, 1.62),
        ("ppo_c8_r6", 8, 6, "PPO (DS)", "ep_block_prob", 0.00, 0.00),
        ("ppo_c8_r6", 8, 6, "PPO (DS)", "ep_accepted_requests", 100.00, 0.00),
        ("ppo_c8_r6", 8, 6, "PPO (DS)", "ep_rejected_requests", 0.00, 0.00),
        ("ppo_c8_r6", 8, 6, "PPO (DS)", "ep_deploy_all", 100.00, 0.00),
        ("ppo_c8_r6", 8, 6, "PPO (DS)", "ep_fdd", 0.00, 0.00),
        ("ppo_c8_r6", 8, 6, "PPO (DS)", "avg_latency", 366.88, 16.71),
        ("ppo_c8_r6", 8, 6, "PPO (DS)", "avg_cost", 7.12, 2.40),
        ("ppo_c8_r6", 8, 6, "PPO (DS)", "avg_cpu_cluster_selected", 19.72, 0.63),
        ("ppo_c8_r6", 8, 6, "PPO (DS)", "gini", 0.79, 0.03),

        ("ppo_c8_r8", 8, 8, "PPO (DS)", "reward", 58.28, 1.65),
        ("ppo_c8_r8", 8, 8, "PPO (DS)", "ep_block_prob", 0.00, 0.00),
        ("ppo_c8_r8", 8, 8, "PPO (DS)", "ep_accepted_requests", 100.00, 0.00),
        ("ppo_c8_r8", 8, 8, "PPO (DS)", "ep_rejected_requests", 0.00, 0.00),
        ("ppo_c8_r8", 8, 8, "PPO (DS)", "ep_deploy_all", 99.27, 0.29),
        ("ppo_c8_r8", 8, 8, "PPO (DS)", "ep_fdd", 0.73, 0.29),
        ("ppo_c8_r8", 8, 8, "PPO (DS)", "avg_latency", 374.73, 18.38),
        ("ppo_c8_r8", 8, 8, "PPO (DS)", "avg_cost", 7.67, 2.46),
        ("ppo_c8_r8", 8, 8, "PPO (DS)", "avg_cpu_cluster_selected", 24.43, 0.75),
        ("ppo_c8_r8", 8, 8, "PPO (DS)", "gini", 0.77, 0.03),

        ("pql_c8_r4", 8, 4, "PQL (MORL)", "ep_accepted_requests", 228.50, 25.21),
        ("pql_c8_r4", 8, 4, "PQL (MORL)", "ep_rejected_requests", 21.50, 25.21),
        ("pql_c8_r4", 8, 4, "PQL (MORL)", "avg_latency", 495.81, 67.80),
        ("pql_c8_r4", 8, 4, "PQL (MORL)", "avg_cost", 32.00, 0.05),
        ("pql_c8_r4", 8, 4, "PQL (MORL)", "avg_cpu_cluster_selected", 18.32, 0.80),
        ("pql_c8_r4", 8, 4, "PQL (MORL)", "gini", 0.88, 0.00),

        ("pql_c8_r6", 8, 6, "PQL (MORL)", "ep_accepted_requests", 250.00, 0.00),
        ("pql_c8_r6", 8, 6, "PQL (MORL)", "ep_rejected_requests", 0.00, 0.00),
        ("pql_c8_r6", 8, 6, "PQL (MORL)", "avg_latency", 469.18, 60.98),
        ("pql_c8_r6", 8, 6, "PQL (MORL)", "avg_cost", 34.24, 4.13),
        ("pql_c8_r6", 8, 6, "PQL (MORL)", "avg_cpu_cluster_selected", 25.02, 2.03),
        ("pql_c8_r6", 8, 6, "PQL (MORL)", "gini", 0.84, 0.03),

        ("pql_c8_r8", 8, 8, "PQL (MORL)", "ep_accepted_requests", 249.80, 0.30),
        ("pql_c8_r8", 8, 8, "PQL (MORL)", "ep_rejected_requests", 0.20, 0.30),
        ("pql_c8_r8", 8, 8, "PQL (MORL)", "avg_latency", 462.28, 58.79),
        ("pql_c8_r8", 8, 8, "PQL (MORL)", "avg_cost", 31.92, 0.07),
        ("pql_c8_r8", 8, 8, "PQL (MORL)", "avg_cpu_cluster_selected", 33.18, 0.92),
        ("pql_c8_r8", 8, 8, "PQL (MORL)", "gini", 0.84, 0.03),
    ]

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Strategy", "Cluster", "Replicas", "Algorithm", "Metric", "Mean", "Std"])

    # Optional: Sort metrics if desired
    metrics_to_plot = ["reward", "ep_block_prob", "ep_deploy_all", "ep_rejected_requests", "avg_latency", "avg_cost", "gini", "avg_cpu_cluster_selected"] # "ep_deploy_all", "ep_fdd",

    # Plot per metric
    sns.set(style="whitegrid")
    for metric in metrics_to_plot:
        plt.figure(figsize=(6, 4))
        df_metric = df[df["Metric"] == metric]

        # Ensure consistent order
        strategy_order = sorted(df_metric["Strategy"].unique(),
                                key=lambda x: (x.split("_")[0], int(x.split("_")[2][1:])))

        ax = sns.barplot(
            data=df_metric,
            x="Strategy",
            y="Mean",
            errorbar=None,
            hue="Algorithm",
            palette="deep",
            order=strategy_order
        )

        # Add error bars manually aligned with bar centers
        for bar, (_, row) in zip(ax.patches, df_metric.set_index("Strategy").loc[strategy_order].iterrows()):
            x = bar.get_x() + bar.get_width() / 2
            y = row["Mean"]
            err = row["Std"]
            plt.errorbar(x=x, y=y, yerr=err, fmt='none', c='black', capsize=5)

        # plt.title(f"Metric: {metric}", fontsize=14)
        plt.ylabel(r'\textbf{' + metric.replace("_", " ").title() + '}', fontsize=20)
        plt.xlabel(r'\textbf{Cluster Configuration}', fontsize=20)
        plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add horizontal gridlines
        plt.legend(title="RL Algorithm", fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{metric}_comparison.pdf", bbox_inches="tight", dpi=250)


if __name__ == '__main__':
    main()
