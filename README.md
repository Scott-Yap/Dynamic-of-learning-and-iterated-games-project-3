# Regret Matching for Discrete Colonel Blotto (S=5, K=3) — Hart Comparison

This repository contains a small, reproducible experiment suite for **two-player zero-sum discrete Colonel Blotto** with payoff
$$
u(x,y)=\frac{1}{K}\sum_{k=1}^K \operatorname{sgn}(x_k-y_k)\in[-1,1],
$$
and a **sampled regret-matching** solver in the style of **Neller & Lanctot** (realised opponent action update).

The core goal is to:
1. Run regret matching self-play to obtain approximate equilibrium behaviour.
2. Measure convergence via **exploitability** and **value**.
3. Compare the learned strategy (after symmetrisation) to the **Hart target random-battlefield marginal** for the specific case \((S,K)=(5,3)\).

> Important: `hart_target_marginal_B553()` is hard-coded for the \((S,K)=(5,3)\) case study. If you change `S` or `K`, the Hart-comparison plots won’t be meaningful unless you implement the corresponding Hart marginal.

---

## Repo layout

- `game.py`  
  Builds the Blotto instance: enumerates actions (allocations) and constructs the payoff matrix `A`.

- `rm.py`  
  **Regret Matching (sampled)** self-play:
  - each time step uses mixed strategies from positive regrets,
  - samples realised actions,
  - updates regrets using the realised opponent move.

- `sym.py`  
  Builds permutation classes of allocations (battlefield labels treated as exchangeable).

- `metrics.py`  
  Evaluation utilities:
  - exploitability / Nash gap,
  - value of mixed strategies,
  - induced random-battlefield marginal,
  - TV distance,
  - Hart target marginal for \((S,K)=(5,3)\).

- `run.py`  
  Main experiment script. Produces figures and prints report-ready summary statistics.

- `figures/`  
  Output directory (generated).

---

## Setup

### 1) Create/activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# .\venv\Scripts\activate # Windows PowerShell
