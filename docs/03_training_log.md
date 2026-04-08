# PPO 训练日志 (Training Log)

记录不同超参数下的收敛情况，防止跑完就忘。

## Experiment 1: Baseline 尝试
* **日期:** 2026-xx-xx
* **核心参数:** lam=0.5, alpha=100.0, reward_scale=1e-3, total_timesteps=100,000
* **收敛情况:** (填写 ep_rew_mean 是否稳定上升，value_loss 是否下降)
* **结果观察:** 智能体是否学会了在 T 时刻前把库存清零？
* **下一步调整:** (比如：如果没清完，尝试调大 alpha；如果前期卖得太快，尝试调小临时冲击参数 eta2)

## Experiment 2: 提高惩罚测试
...