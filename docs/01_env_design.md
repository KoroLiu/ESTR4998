# RL 环境设计 (MDP 建模)

本文件基于报告第3节和第4节的理论推导，将其映射为强化学习的标准马尔可夫决策过程 (MDP)。

## 1. 状态空间 (State Space)
智能体在每个时间步 $t_n$ 需要观察到的市场和自身状态：
* **剩余时间 (Time Remaining):** $T - t_n$ 或者归一化的时间步比例。
* **剩余库存 (Inventory):** $q_n$。为方便神经网络学习，可归一化为 $q_n / q_0 \in [0, 1]$。
* **当前不受影响的市场价格 (Unaffected Price):** $S_n$。遵循几何布朗运动 (GBM)。

## 2. 动作空间 (Action Space)
* **动作:** 卖出速率 $v_n \ge 0$。
* **工程处理:** 神经网络通常输出 $[-1, 1]$ 的连续值。我们可以将其映射为实际卖出速率，或者直接将其定义为“当前剩余库存的清仓比例”。为了严格对标报告中的公式 (46) $q_{n+1} = q_n - v_n \Delta t$，我们将动作映射为具体的速率 $v_n$。

## 3. 状态转移 (State Transition)
根据报告离散化形式 (Eq 46):
* **库存更新:** $q_{n+1} = \max(0, q_n - v_n \Delta t)$。
* **价格更新:** 采用精确的 GBM 离散解以保证数值稳定：
    $$S_{n+1} = S_n \exp\left(-\frac{1}{2}\sigma^2 \Delta t + \sigma \sqrt{\Delta t} Z\right)$$
    其中 $Z \sim \mathcal{N}(0,1)$。

## 4. 奖励函数 (Reward Function)
强化学习的目标是最大化累积奖励，这等价于报告中最小化执行成本与风险惩罚的组合。
根据报告公式 (19) 和离散化公式 (47)：
* **单步奖励 (Step Reward):**
    $$R_n = - \left[ (f(q_0 - q_n) + g(v_n)) v_n \Delta t + \frac{\lambda}{2} \sigma^2 S_n^2 q_n^2 \Delta t \right]$$
* **终局惩罚 (Terminal Penalty):** 如果在 $T$ 时刻库存没卖完，给予极大的惩罚：
    $$R_N = - \frac{\alpha}{2} S_N^2 q_N^2$$