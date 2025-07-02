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


def process_file(filepath, test_name, full_data):
    data = pd.read_csv(filepath)

    # Assign the file-based test name
    data['test'] = test_name

    # Extract algorithm, CPU, and RAM using regex
    data[['algorithm', 'cluster', 'replicas']] = data['test'].str.extract(r'(\w+)_c(\d+)_r(\d+)')

    # Convert CPU and RAM to numeric (optional, if needed for plots or sorting)
    data['cluster'] = data['cluster'].astype(int)
    data['replicas'] = data['replicas'].astype(int)
    data['config'] = 'c' + data['cluster'].astype(str) + '_r' + data['replicas'].astype(str)

    data['acceptance_rate'] = data['ep_accepted_requests'] / (
            data['ep_accepted_requests'] + data['ep_rejected_requests'])

    # Save data in a df to save all data together
    full_data.append(data)

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
    dir = 'results/power/'
    output_dir = dir + 'processed/'
    POWER=True

    full_data = []
    for filename in os.listdir(dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(dir, filename)
            test_name = os.path.splitext(filename)[0]  # Remove .csv
            process_file(filepath, test_name, full_data)

    # Save data csv
    os.makedirs(output_dir, exist_ok=True)
    full_df = pd.concat(full_data, ignore_index=True)
    full_df.to_csv(os.path.join(output_dir, "full.csv"), index=False)

    # Load your CSV file
    df = pd.read_csv(output_dir + "full.csv")

    if not POWER:
        hue_order = ['DQN', 'PPO', 'PQL', 'Geometric', 'MPMOQ']  # ['DQN', 'PPO', 'PQL', 'Geometric', 'MPMOQ']
        # Generate the 'deep' palette with, say, 8 colors
        palette = sns.color_palette("deep", 8)

    else:
        hue_order = ['PQL', 'Geometric', 'MPMOQ']
        # Generate the 'deep' palette with, say, 8 colors
        palette = sns.color_palette("deep", 8)

        # Skip the first two colors
        palette = palette[2:]

    '''
    plt.figure(figsize=(7, 5))
    # Plot the CDF of avg_latency
    ax = sns.ecdfplot(data=df, x="avg_latency", hue="algorithm", palette="deep")

    # Customize the plot
    plt.ylabel(r'\textbf{CDF}', fontsize=20)
    plt.xlabel(r'\textbf{Latency [ms]}', fontsize=20)

    # Access and customize the legend from the axes object
    ax.legend_.set_title("RL Algorithm")
    for text in ax.legend_.get_texts():
        text.set_fontsize(14)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("seaborn_cdf_latency.pdf", bbox_inches="tight", dpi=250)
    '''

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=df,
        x="config",
        y="avg_latency",
        errorbar='ci',
        hue="algorithm",
        palette=palette,
        hue_order=hue_order,
    )
    plt.xlabel(r'\textbf{Cluster Configuration}', fontsize=20)
    plt.ylabel(r'\textbf{Latency [ms]}', fontsize=20)
    plt.legend(title="RL Algorithm", fontsize=12, title_fontsize=14, ncols=3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("seaborn_latency.pdf", bbox_inches="tight", dpi=250)

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=df,
        x="config",
        y="avg_cost",
        errorbar='ci',
        hue="algorithm",
        palette=palette,
        hue_order=hue_order,
    )
    plt.xlabel(r'\textbf{Cluster Configuration}', fontsize=20)
    plt.ylabel(r'\textbf{Cost [units]}', fontsize=20)
    plt.legend(title="RL Algorithm", fontsize=12, title_fontsize=14, ncols=3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("seaborn_cost.pdf", bbox_inches="tight", dpi=250)

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=df,
        x="config",
        y="gini",
        errorbar='ci',
        hue="algorithm",
        palette=palette,
        hue_order=hue_order,
    )
    plt.xlabel(r'\textbf{Cluster Configuration}', fontsize=20)
    plt.ylabel(r'\textbf{Gini [0-1]}', fontsize=20)
    plt.legend(title="RL Algorithm", fontsize=12, title_fontsize=14, ncols=3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("seaborn_gini.pdf", bbox_inches="tight", dpi=250)

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=df,
        x="config",
        y="acceptance_rate",
        errorbar='ci',
        hue="algorithm",
        palette=palette,
        hue_order=hue_order,
    )
    plt.xlabel(r'\textbf{Cluster Configuration}', fontsize=20)
    plt.ylabel(r'\textbf{Acceptance Rate [0-1]}', fontsize=20)
    plt.legend(title="RL Algorithm", fontsize=12, title_fontsize=14, ncols=3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("seaborn_acceptance_rate.pdf", bbox_inches="tight", dpi=250)

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=df,
        x="config",
        y="avg_cpu_cluster_selected",
        errorbar='ci',
        hue="algorithm",
        palette=palette,
        hue_order=hue_order,
    )
    plt.xlabel(r'\textbf{Cluster Configuration}', fontsize=20)
    plt.ylabel(r'\textbf{CPU Usage [\%]}', fontsize=20)
    plt.legend(title="RL Algorithm", fontsize=12, title_fontsize=14, ncols=3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("seaborn_cpu.pdf", bbox_inches="tight", dpi=250)

    plt.figure(figsize=(7, 5))
    sns.violinplot(
        data=df,
        x="config",
        y="avg_latency",
        # errorbar='ci',
        hue="algorithm",
        palette=palette,
        linewidth=1,
        inner_kws=dict(box_width=4, whis_width=0.2, color=".8"),
        hue_order=hue_order,
    )
    plt.xlabel(r'\textbf{Cluster Configuration}', fontsize=20)
    plt.ylabel(r'\textbf{Latency [ms]}', fontsize=20)
    plt.legend(title="RL Algorithm", fontsize=12, title_fontsize=14, ncols=3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("seaborn_violin_latency.pdf", bbox_inches="tight", dpi=250)

    plt.figure(figsize=(7, 5))
    sns.violinplot(
        data=df,
        x="config",
        y="avg_cost",
        # errorbar='ci',
        hue="algorithm",
        palette=palette,
        linewidth=1,
        inner_kws=dict(box_width=4, whis_width=0.2, color=".8"),
        hue_order=hue_order,
    )
    plt.xlabel(r'\textbf{Cluster Configuration}', fontsize=20)
    plt.ylabel(r'\textbf{Cost [units]}', fontsize=20)
    plt.legend(title="RL Algorithm", fontsize=12, title_fontsize=14, ncols=3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("seaborn_violin_cost.pdf", bbox_inches="tight", dpi=250)

    plt.figure(figsize=(7, 5))
    sns.violinplot(
        data=df,
        x="config",
        y="gini",
        # errorbar='ci',
        hue="algorithm",
        palette=palette,
        linewidth=1,
        inner_kws=dict(box_width=4, whis_width=0.2, color=".8"),
        hue_order=hue_order,
    )
    plt.xlabel(r'\textbf{Cluster Configuration}', fontsize=20)
    plt.ylabel(r'\textbf{Gini [0-1]}', fontsize=20)
    plt.legend(title="RL Algorithm", fontsize=12, title_fontsize=14, ncols=3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("seaborn_violin_gini.pdf", bbox_inches="tight", dpi=250)

    if POWER:
        plt.figure(figsize=(7, 5))
        sns.violinplot(
            data=df,
            x="config",
            y="avg_power",
            # errorbar='ci',
            hue="algorithm",
            palette=palette,
            linewidth=1,
            inner_kws=dict(box_width=4, whis_width=0.2, color=".8"),
            hue_order=hue_order,
        )
        plt.xlabel(r'\textbf{Cluster Configuration}', fontsize=20)
        plt.ylabel(r'\textbf{Power Consumption [W]}', fontsize=20)
        plt.legend(title="RL Algorithm", fontsize=12, title_fontsize=14, ncols=3)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("seaborn_violin_power.pdf", bbox_inches="tight", dpi=250)

        plt.figure(figsize=(7, 5))
        sns.set_context("talk")
        ax = sns.pairplot(df,
                          hue="algorithm",
                          vars=["avg_latency", "avg_cost", "avg_power", "acceptance_rate", "gini"],
                          markers=["o", "s", "v", "^", "D"],
                          palette=palette,
                          hue_order=hue_order,
                          # corner=True,
                          # kind="hist",
                          )

        ax._legend.set_title("RL Algorithm")
        ax._legend.get_title().set_fontsize(14)  # Correct way to set title font size

        ax._legend.set_bbox_to_anchor((1.05, 0.5))  # Shift right
        ax._legend.set_loc("center left")  # Anchor on left of the legend box
        # Optional: adjust font sizes
        # ax._legend.set_title_fontsize(14)
        for text in ax._legend.texts:
            text.set_fontsize(12)

        # plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("seaborn_pairplot.pdf", bbox_inches="tight", dpi=250)

    else:
        plt.figure(figsize=(7, 5))
        sns.set_context("talk")
        ax = sns.pairplot(df,
                          hue="algorithm",
                          vars=["avg_latency", "avg_cost", "acceptance_rate", "gini"],
                          markers=["o", "s", "v", "^", "D"],
                          palette=palette,
                          hue_order=hue_order,
                          # corner=True,
                          # kind="hist",
                          )

        ax._legend.set_title("RL Algorithm")
        ax._legend.get_title().set_fontsize(14)  # Correct way to set title font size

        ax._legend.set_bbox_to_anchor((1.05, 0.5))  # Shift right
        ax._legend.set_loc("center left")  # Anchor on left of the legend box
        # Optional: adjust font sizes
        # ax._legend.set_title_fontsize(14)
        for text in ax._legend.texts:
            text.set_fontsize(12)

        # plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("seaborn_pairplot.pdf", bbox_inches="tight", dpi=250)

    '''
    # Data
    data = [
        # Format: (agent_config, metric_name, mean, std)
        ("dqn_c8_r4", "c8_r4", 8, 4, "DQN (DS)", "reward", 56.70, 2.70),
        ("dqn_c8_r4", "c8_r4", 8, 4, "DQN (DS)", "ep_block_prob", 0.02, 0.03),
        ("dqn_c8_r4", "c8_r4", 8, 4, "DQN (DS)", "ep_accepted_requests", 98.80, 1.37),
        ("dqn_c8_r4", "c8_r4", 8, 4, "DQN (DS)", "ep_rejected_requests", 1.20, 1.37),
        ("dqn_c8_r4", "c8_r4", 8, 4, "DQN (DS)", "ep_deploy_all", 91.00, 7.32),
        ("dqn_c8_r4", "c8_r4", 8, 4, "DQN (DS)", "ep_fdd", 6.60, 4.69),
        ("dqn_c8_r4", "c8_r4", 8, 4, "DQN (DS)", "avg_latency", 455.42, 12.12),
        ("dqn_c8_r4", "c8_r4", 8, 4, "DQN (DS)", "avg_cost", 10.31, 2.40),
        ("dqn_c8_r4", "c8_r4", 8, 4, "DQN (DS)", "avg_cpu", 13.31, 0.63),
        ("dqn_c8_r4", "c8_r4", 8, 4, "DQN (DS)", "gini", 0.65, 0.04),

        ("dqn_c8_r6", "c8_r6", 8, 6, "DQN (DS)", "reward", 55.84, 3.25),
        ("dqn_c8_r6", "c8_r6", 8, 6, "DQN (DS)", "ep_block_prob", 0.02, 0.02),
        ("dqn_c8_r6", "c8_r6", 8, 6, "DQN (DS)", "ep_accepted_requests", 98.37, 1.64),
        ("dqn_c8_r6", "c8_r6", 8, 6, "DQN (DS)", "ep_rejected_requests", 1.63, 1.64),
        ("dqn_c8_r6", "c8_r6", 8, 6, "DQN (DS)", "ep_deploy_all", 89.57, 7.52),
        ("dqn_c8_r6", "c8_r6", 8, 6, "DQN (DS)", "ep_fdd", 8.80, 5.94),
        ("dqn_c8_r6", "c8_r6", 8, 6, "DQN (DS)", "avg_latency", 448.02, 11.91),
        ("dqn_c8_r6", "c8_r6", 8, 6, "DQN (DS)", "avg_cost", 11.22, 2.43),
        ("dqn_c8_r6", "c8_r6", 8, 6, "DQN (DS)", "avg_cpu", 17.82, 1.07),
        ("dqn_c8_r6", "c8_r6", 8, 6, "DQN (DS)", "gini", 0.65, 0.03),

        ("dqn_c8_r8", "c8_r8", 8, 8, "DQN (DS)", "reward", 55.78, 2.52),
        ("dqn_c8_r8", "c8_r8", 8, 8, "DQN (DS)", "ep_block_prob", 0.01, 0.01),
        ("dqn_c8_r8", "c8_r8", 8, 8, "DQN (DS)", "ep_accepted_requests", 98.53, 1.13),
        ("dqn_c8_r8", "c8_r8", 8, 8, "DQN (DS)", "ep_rejected_requests", 1.47, 1.13),
        ("dqn_c8_r8", "c8_r8", 8, 8, "DQN (DS)", "ep_deploy_all", 85.57, 8.04),
        ("dqn_c8_r8", "c8_r8", 8, 8, "DQN (DS)", "ep_fdd", 12.97, 6.98),
        ("dqn_c8_r8", "c8_r8", 8, 8, "DQN (DS)", "avg_latency", 442.93, 15.03),
        ("dqn_c8_r8", "c8_r8", 8, 8, "DQN (DS)", "avg_cost", 12.07, 2.23),
        ("dqn_c8_r8", "c8_r8", 8, 8, "DQN (DS)", "avg_cpu", 21.18, 1.40),
        ("dqn_c8_r8", "c8_r8", 8, 8, "DQN (DS)", "gini", 0.66, 0.03),

        # --- Geometric entries ---
        ("geo_c8_r4", "c8_r4", 8, 4, "Geometric (MORL)", "ep_accepted_requests", 242.30, 7.74),
        ("geo_c8_r4", "c8_r4", 8, 4, "Geometric (MORL)", "ep_rejected_requests", 7.70, 7.74),
        ("geo_c8_r4", "c8_r4", 8, 4, "Geometric (MORL)", "avg_latency", 576.33, 65.50),
        ("geo_c8_r4", "c8_r4", 8, 4, "Geometric (MORL)", "avg_cost", 38.30, 5.48),
        ("geo_c8_r4", "c8_r4", 8, 4, "Geometric (MORL)", "avg_cpu", 19.08, 1.44),
        ("geo_c8_r4", "c8_r4", 8, 4, "Geometric (MORL)", "gini", 0.52, 0.11),

        ("geo_c8_r6", "c8_r6", 8, 6, "Geometric (MORL)", "ep_accepted_requests", 222.10, 23.33),
        ("geo_c8_r6", "c8_r6", 8, 6, "Geometric (MORL)", "ep_rejected_requests", 27.90, 23.33),
        ("geo_c8_r6", "c8_r6", 8, 6, "Geometric (MORL)", "avg_latency", 549.75, 65.66),
        ("geo_c8_r6", "c8_r6", 8, 6, "Geometric (MORL)", "avg_cost", 41.18, 3.49),
        ("geo_c8_r6", "c8_r6", 8, 6, "Geometric (MORL)", "avg_cpu", 22.85, 2.77),
        ("geo_c8_r6", "c8_r6", 8, 6, "Geometric (MORL)", "gini", 0.53, 0.11),

        ("geo_c8_r8", "c8_r8", 8, 8, "Geometric (MORL)", "ep_accepted_requests", 228.90, 5.46),
        ("geo_c8_r8", "c8_r8", 8, 8, "Geometric (MORL)", "ep_rejected_requests", 21.10, 5.46),
        ("geo_c8_r8", "c8_r8", 8, 8, "Geometric (MORL)", "avg_latency", 656.93, 43.64),
        ("geo_c8_r8", "c8_r8", 8, 8, "Geometric (MORL)", "avg_cost", 22.70, 5.90),
        ("geo_c8_r8", "c8_r8", 8, 8, "Geometric (MORL)", "avg_cpu", 33.23, 4.61),
        ("geo_c8_r8", "c8_r8", 8, 8, "Geometric (MORL)", "gini", 0.51, 0.09),

        # --- MPMOQ (corrected) ---
        ("mpmoq_c8_r4", "c8_r4", 8, 4, "MPMOQ (MORL)", "ep_accepted_requests", 219.70, 1.63),
        ("mpmoq_c8_r4", "c8_r4", 8, 4, "MPMOQ (MORL)", "ep_rejected_requests", 30.30, 1.63),
        ("mpmoq_c8_r4", "c8_r4", 8, 4, "MPMOQ (MORL)", "avg_latency", 644.82, 2.69),
        ("mpmoq_c8_r4", "c8_r4", 8, 4, "MPMOQ (MORL)", "avg_cost", 9.67, 0.07),
        ("mpmoq_c8_r4", "c8_r4", 8, 4, "MPMOQ (MORL)", "avg_cpu", 13.22, 0.29),
        ("mpmoq_c8_r4", "c8_r4", 8, 4, "MPMOQ (MORL)", "gini", 0.17, 0.01),

        ("mpmoq_c8_r6", "c8_r6", 8, 6, "MPMOQ (MORL)", "ep_accepted_requests", 227.83, 1.70),
        ("mpmoq_c8_r6", "c8_r6", 8, 6, "MPMOQ (MORL)", "ep_rejected_requests", 22.17, 1.70),
        ("mpmoq_c8_r6", "c8_r6", 8, 6, "MPMOQ (MORL)", "avg_latency", 677.24, 4.10),
        ("mpmoq_c8_r6", "c8_r6", 8, 6, "MPMOQ (MORL)", "avg_cost", 27.67, 0.13),
        ("mpmoq_c8_r6", "c8_r6", 8, 6, "MPMOQ (MORL)", "avg_cpu", 14.63, 0.37),
        ("mpmoq_c8_r6", "c8_r6", 8, 6, "MPMOQ (MORL)", "gini", 0.16, 0.02),

        ("mpmoq_c8_r8", "c8_r8", 8, 8, "MPMOQ (MORL)", "ep_accepted_requests", 239.30, 0.98),
        ("mpmoq_c8_r8", "c8_r8", 8, 8, "MPMOQ (MORL)", "ep_rejected_requests", 10.70, 0.98),
        ("mpmoq_c8_r8", "c8_r8", 8, 8, "MPMOQ (MORL)", "avg_latency", 688.26, 2.84),
        ("mpmoq_c8_r8", "c8_r8", 8, 8, "MPMOQ (MORL)", "avg_cost", 31.91, 0.24),
        ("mpmoq_c8_r8", "c8_r8", 8, 8, "MPMOQ (MORL)", "avg_cpu", 18.04,  0.48),
        ("mpmoq_c8_r8", "c8_r8", 8, 8, "MPMOQ (MORL)", "gini", 0.22, 0.02),

        ("ppo_c8_r4", "c8_r4", 8, 4, "PPO (DS)", "reward",  57.90, 1.40),
        ("ppo_c8_r4", "c8_r4", 8, 4, "PPO (DS)", "ep_block_prob", 0.00, 0.00),
        ("ppo_c8_r4", "c8_r4", 8, 4, "PPO (DS)", "ep_accepted_requests", 100.00, 0.00),
        ("ppo_c8_r4", "c8_r4", 8, 4, "PPO (DS)", "ep_rejected_requests", 0.00, 0.00),
        ("ppo_c8_r4", "c8_r4", 8, 4, "PPO (DS)", "ep_deploy_all", 100.00, 0.00),
        ("ppo_c8_r4", "c8_r4", 8, 4, "PPO (DS)", "ep_fdd", 0.00, 0.00),
        ("ppo_c8_r4", "c8_r4", 8, 4, "PPO (DS)", "avg_latency", 365.23, 16.45),
        ("ppo_c8_r4", "c8_r4", 8, 4, "PPO (DS)", "avg_cost", 6.94, 2.38),
        ("ppo_c8_r4", "c8_r4", 8, 4, "PPO (DS)", "avg_cpu", 14.02, 0.53),
        ("ppo_c8_r4", "c8_r4", 8, 4, "PPO (DS)", "gini", 0.81, 0.02),

        ("ppo_c8_r6", "c8_r6", 8, 6, "PPO (DS)", "reward", 58.31, 1.62),
        ("ppo_c8_r6", "c8_r6", 8, 6, "PPO (DS)", "ep_block_prob", 0.00, 0.00),
        ("ppo_c8_r6", "c8_r6", 8, 6, "PPO (DS)", "ep_accepted_requests", 100.00, 0.00),
        ("ppo_c8_r6", "c8_r6", 8, 6, "PPO (DS)", "ep_rejected_requests", 0.00, 0.00),
        ("ppo_c8_r6", "c8_r6", 8, 6, "PPO (DS)", "ep_deploy_all", 100.00, 0.00),
        ("ppo_c8_r6", "c8_r6", 8, 6, "PPO (DS)", "ep_fdd", 0.00, 0.00),
        ("ppo_c8_r6", "c8_r6", 8, 6, "PPO (DS)", "avg_latency", 366.88, 16.71),
        ("ppo_c8_r6", "c8_r6", 8, 6, "PPO (DS)", "avg_cost", 7.12, 2.40),
        ("ppo_c8_r6", "c8_r6", 8, 6, "PPO (DS)", "avg_cpu", 19.72, 0.63),
        ("ppo_c8_r6", "c8_r6", 8, 6, "PPO (DS)", "gini", 0.79, 0.03),

        ("ppo_c8_r8", "c8_r8", 8, 8, "PPO (DS)", "reward", 58.28, 1.65),
        ("ppo_c8_r8", "c8_r8", 8, 8, "PPO (DS)", "ep_block_prob", 0.00, 0.00),
        ("ppo_c8_r8", "c8_r8", 8, 8, "PPO (DS)", "ep_accepted_requests", 100.00, 0.00),
        ("ppo_c8_r8", "c8_r8", 8, 8, "PPO (DS)", "ep_rejected_requests", 0.00, 0.00),
        ("ppo_c8_r8", "c8_r8", 8, 8, "PPO (DS)", "ep_deploy_all", 99.27, 0.29),
        ("ppo_c8_r8", "c8_r8", 8, 8, "PPO (DS)", "ep_fdd", 0.73, 0.29),
        ("ppo_c8_r8", "c8_r8", 8, 8, "PPO (DS)", "avg_latency", 374.73, 18.38),
        ("ppo_c8_r8", "c8_r8", 8, 8, "PPO (DS)", "avg_cost", 7.67, 2.46),
        ("ppo_c8_r8", "c8_r8", 8, 8, "PPO (DS)", "avg_cpu", 24.43, 0.75),
        ("ppo_c8_r8", "c8_r8", 8, 8, "PPO (DS)", "gini", 0.77, 0.03),

        ("pql_c8_r4", "c8_r4", 8, 4, "PQL (MORL)", "ep_accepted_requests", 228.50, 25.21),
        ("pql_c8_r4", "c8_r4", 8, 4, "PQL (MORL)", "ep_rejected_requests", 21.50, 25.21),
        ("pql_c8_r4", "c8_r4", 8, 4, "PQL (MORL)", "avg_latency", 495.81, 67.80),
        ("pql_c8_r4", "c8_r4", 8, 4, "PQL (MORL)", "avg_cost", 32.00, 0.05),
        ("pql_c8_r4", "c8_r4", 8, 4, "PQL (MORL)", "avg_cpu", 18.32, 0.80),
        ("pql_c8_r4", "c8_r4", 8, 4, "PQL (MORL)", "gini", 0.88, 0.00),

        ("pql_c8_r6", "c8_r6", 8, 6, "PQL (MORL)", "ep_accepted_requests", 250.00, 0.00),
        ("pql_c8_r6", "c8_r6", 8, 6, "PQL (MORL)", "ep_rejected_requests", 0.00, 0.00),
        ("pql_c8_r6", "c8_r6", 8, 6, "PQL (MORL)", "avg_latency", 469.18, 60.98),
        ("pql_c8_r6", "c8_r6", 8, 6, "PQL (MORL)", "avg_cost", 34.24, 4.13),
        ("pql_c8_r6", "c8_r6", 8, 6, "PQL (MORL)", "avg_cpu", 25.02, 2.03),
        ("pql_c8_r6", "c8_r6", 8, 6, "PQL (MORL)", "gini", 0.84, 0.03),

        ("pql_c8_r8", "c8_r8", 8, 8, "PQL (MORL)", "ep_accepted_requests", 249.80, 0.30),
        ("pql_c8_r8", "c8_r8", 8, 8, "PQL (MORL)", "ep_rejected_requests", 0.20, 0.30),
        ("pql_c8_r8", "c8_r8", 8, 8, "PQL (MORL)", "avg_latency", 462.28, 58.79),
        ("pql_c8_r8", "c8_r8", 8, 8, "PQL (MORL)", "avg_cost", 31.92, 0.07),
        ("pql_c8_r8", "c8_r8", 8, 8, "PQL (MORL)", "avg_cpu", 33.18, 0.92),
        ("pql_c8_r8", "c8_r8", 8, 8, "PQL (MORL)", "gini", 0.84, 0.03),
    ]

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Strategy", "Config", "Cluster", "Replicas", "Algorithm", "Metric", "Mean", "Std"])

    
    # Optional: Sort metrics if desired
    metrics_to_plot = ["reward", "ep_block_prob", "ep_deploy_all", "ep_rejected_requests", "avg_latency", "avg_cost",
                       "gini", "avg_cpu"]  # "ep_deploy_all", "ep_fdd",

    # Plot per metric
    sns.set(style="whitegrid")
    for metric in metrics_to_plot:
        plt.figure(figsize=(7, 5))
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
    

    # Metrics to plot
    metrics_to_plot = ["reward", "ep_block_prob", "ep_deploy_all", "ep_rejected_requests", "avg_latency", "avg_cost",
                       "gini", "avg_cpu"] # "reward", "ep_block_prob", "ep_deploy_all",

    # Load into a pandas DataFrame
    df = pd.DataFrame(data, columns=[
            "strategy", "config", "cluster", "replicas",
            "algorithm", "metric", "mean", "std"
    ])

    # Filter for only those metrics
    df_filtered = df[df["metric"].isin(metrics_to_plot)]

    # Pivot the data for better plotting
    pivoted = df_filtered.pivot_table(
            index=["config", "algorithm"],
            columns="metric",
            values=["mean", "std"],
    ).reset_index()
    print(pivoted.head())

    # Plotting
    for metric in metrics_to_plot:
        print("----------------------------------------------------------------")
        print(metric)

        plt.figure(figsize=(7, 5))
        ax = sns.barplot(
                data=df_filtered[df_filtered["metric"] == metric],
                x="config",
                y="mean",
                hue="algorithm",
                errorbar=None,
                # errwidth="std",
                palette="deep",
        )

        # plt.title(r'\textbf{Comparison of '+ metric.replace('_', ' ').title() + ' across Algorithms and Cluster Configurations', fontsize=20)
        plt.ylabel(r'\textbf{' + metric.replace("_", " ").title() + '}', fontsize=20)
        plt.xlabel(r'\textbf{Cluster Configuration}', fontsize=20)
        plt.legend(title="RL Algorithm", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{metric}_comparison.pdf", bbox_inches="tight", dpi=250)
    '''


if __name__ == '__main__':
    main()
