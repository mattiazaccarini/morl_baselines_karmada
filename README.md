# GaiaRL Baselines for Karmada

Code repository for the paper:

**"GaiaRL: Multi-Objective RL for Sustainable VNF Placement in the Compute Continuum"** (submitted to NOMS 2026).

This repository provides runnable scripts for multi-objective reinforcement learning experiments on Karmada scheduling environments, including both **with-power** and **no-power** variants.

## System Requirements

- **CPU**: Modern multi-core CPU
- **RAM**: 8 GB minimum recommended
- **Storage**:
  - `< 15-20 GB` for code + dependencies + simulations
  - typically `< 10 GB` if GPU is not used
- **GPU**: Not required
- **Specialized hardware**: None

## Installation

Estimated setup time: **5-15 minutes** (depends on machine performance and package installation speed).

From the repository root:

```bash
pip install -r requirements.txt
```

The main Python dependencies are managed through `requirements.txt` (e.g., `gymnasium`, `mo-gymnasium`, `morl-baselines`, `wandb`, `numpy`, `scipy`, `pandas`, `matplotlib`).

## How to Run

Estimated execution time: **2-20 minutes** per run (depends on algorithm, hardware, and configuration).

1. Move to the experiment folder:
   ```bash
   cd gym-multi-k8s
   ```

2. To retrieve **Multi-Objective Reinforcement Learning (MORL)** metrics, run one of the NOMS scripts below using `python3 -m ...`.

### NOMS Scripts (No-Power Variants)

- **PQL (no-power)**:
  ```bash
  python3 -m run_noms_algorithms.run_noms_pql_nopower
  ```

- **Geometric PQL (no-power)**:
  ```bash
  python3 -m run_noms_algorithms.run_noms_geometric_pql_nopower
  ```

- **MPMOQ-Learning (no-power)**:
  ```bash
  python3 -m run_noms_algorithms.run_noms_mpmoqlearning_nopower
  ```

### NOMS Scripts (Power-Aware Variants)

- **PQL (power-aware)**:
  ```bash
  python3 -m run_noms_algorithms.run_noms_pql
  ```

- **Geometric PQL (power-aware)**:
  ```bash
  python3 -m run_noms_algorithms.run_noms_geometric_pql
  ```

- **MPMOQ-Learning (power-aware)**:
  ```bash
  python3 -m run_noms_algorithms.run_noms_mpmoqlearning
  ```

3. To retrieve **DQN/PPO DeepSets** metrics, run:

```bash
python3 run_fgcs.py
```

Default parameters are included to enable a first run. You can also provide custom parameters according to your needs, for example:

- **DQN DeepSets (explicit parameters)**:
  ```bash
  python3 run_fgcs.py --alg dqn_deepsets --env_name karmada --num_clusters 4 --min_replicas 1 --max_replicas 16
  ```

- **PPO DeepSets**:
  ```bash
  python3 run_fgcs.py --alg ppo_deepsets --env_name karmada --num_clusters 4 --min_replicas 1 --max_replicas 16
  ```

## Notes

- Run commands from inside `gym-multi-k8s` so Python can resolve local modules correctly.
- Several scripts enable Weights & Biases logging (`wandb`). If needed, configure your environment (e.g., login or offline mode) before running.
- Generated CSV results and run artifacts are written in the experiment directory.

## Repository Structure (Relevant Paths)

- `requirements.txt`
- `gym-multi-k8s/run_noms_algorithms/`
- `gym-multi-k8s/envs/`
