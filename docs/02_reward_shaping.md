# 奖励塑形与超参数设置

## 冲击函数 (Impact Functions)
报告采用了二次函数形式 (Eq 30)：
* **永久冲击:** $f(Q_t) = \gamma_1 Q_t + \gamma_2 Q_t^2$
* **临时冲击:** $g(v_t) = \eta_1 v_t + \eta_2 v_t^2$

## 训练难点与 Reward Scaling
在 PPO 训练中，由于金融数据的绝对值很大（例如 $S_t \approx 100$），导致 Reward 的绝对值可能非常大（负几千），这会让 PPO 的 Critic 很难收敛。
**解决策略:** 1. 在环境中对 Reward 除以一个常数（例如 $10^4$）进行缩放 (Reward Scaling)。
2. 终局惩罚系数 $\alpha$ 必须足够大，以迫使智能体学会清仓，但太大又会导致梯度爆炸。建议在代码中将 $\alpha$ 设置为动态可调，初始设为 100 左右。