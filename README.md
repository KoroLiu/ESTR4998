# RL_Liquidation_FYP: Optimal Liquidation with GBM and Nonlinear Impact

[cite_start]This project explores the optimal liquidation problem using Reinforcement Learning (PPO), based on an extension of the classical Almgren-Chriss (AC) model[cite: 1, 13]. [cite_start]Our research incorporates Geometric Brownian Motion (GBM) for price dynamics and nonlinear functional forms for both permanent and temporary price impacts[cite: 19, 106].

## 👥 Authors
- [cite_start]**Shuo Huang**: Department of Mathematics / Information Engineering, CUHK [cite: 2, 3, 4]
- [cite_start]**Zhixu Liu**: Department of Systems Engineering and Engineering Management, CUHK [cite: 5, 6, 7]

## 📖 Project Background
[cite_start]The core objective is to liquidate a large position within a finite time horizon while balancing execution costs (market impact) and market risk (volatility)[cite: 10]. [cite_start]We utilize a **Mean-Quadratic-Variation** criterion as the objective function to ensure time-consistency[cite: 126, 131].

## 📁 Project Structure
- [cite_start]`Environment/`: Custom Gymnasium environment for the liquidation MDP[cite: 170, 175].
- `RL_Training/`: PPO training scripts using Stable Baselines3.
- [cite_start]`PDE_Baseline/`: Numerical solutions based on Hamilton-Jacobi-Bellman (HJB) equations (Work in progress)[cite: 146, 175].
- `Evaluation/`: Backtesting scripts and parameter sensitivity analysis (Figure 1-3 replication).
- `docs/`: Design logs, reward shaping strategies, and report drafts.
- `models/`: Saved trained PPO model weights (`.zip` files).

## 🚀 Getting Started

### 1. Setup Environment
Ensure you have Anaconda or Miniconda installed in a Linux/WSL environment.
```bash
conda activate finrl
pip install -r requirements.txt

2. Training the AgentOpen RL_Training/train_ppo.ipynb and run all cells. This will train "expert models" for different risk-aversion parameters ($\lambda$).3. EvaluationOpen Evaluation/backtest_analysis.ipynb to load trained models and generate comparative plots for:Inventory Trajectories ($q_t$)Selling Rates ($v_t$)Execution Price Paths ($\tilde{S}_t$)🛠️ Implementation DetailsAlgorithm: Proximal Policy Optimization (PPO).Framework: Stable Baselines3 / PyTorch.Key Equation: Reward function derived from the expected infinitesimal change in wealth net of a running penalty on its quadratic variation.