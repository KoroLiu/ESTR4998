# 05: Simulation Environments & Data Architecture

## 1. Overview (项目仿真总览)
本项目旨在研究最优清算问题 (Optimal Liquidation)。为了严谨地验证强化学习 (RL) 在复杂金融微观结构中的优越性，我们设计了两个层级的仿真环境：
1. **Analytical Baseline Env (基于传统 AC 模型的解析环境)**：用于对齐和验证传统的 PDE/DP 理论基线。
2. **Microstructure LOB Env (基于限价单簿的微观结构环境)**：用于突破传统 PDE 无法求解的维度灾难，展现 RL (尤其是时序网络 LSTM) 在弹性流动性市场中的真正实力。

---

## 2. Environment 1: Analytical Baseline (liquidation_env.py)
此环境是经典的 Almgren-Chriss 模型的扩展，完全基于连续时间的随机微分方程 (SDE) 进行离散化仿真。

### 2.1 仿真数据类型 (Data Simulated)
* **价格生成 (Price Dynamics)**: 使用几何布朗运动 (Geometric Brownian Motion, GBM) 模拟无下限的资产价格轨迹。无历史记忆，严格满足马尔可夫性 (Markov Property)。
* **冲击模型 (Impact Model)**: 
  * 永久冲击 (Permanent Impact): `f(Q) = gamma1*Q + gamma2*Q^2`
  * 临时冲击 (Temporary Impact): `g(v) = eta1*v + eta2*v^2`

### 2.2 MDP 定义 (Markov Decision Process)
* **State Space (状态空间)**: `[t_norm, q_norm, S_norm]` (当前时间进度, 剩余库存比例, 归一化价格)。维度：3。
* **Action Space (动作空间)**: `[-1, 1]` 连续动作，映射到实际的卖出速率 `v_t`。
* **Reward (奖励函数)**: 每一单步的执行成本 (Execution Cost) 与二次变差风险惩罚 (Quadratic Variation Risk Penalty) 的负和。

---

## 3. Environment 2: Microstructure LOB (liquidation_env_lob.py)
此环境打破了传统 PDE 的“真空”假设，引入了具有物理意义的订单簿 (Limit Order Book) 机制和流动性恢复机制 (Resilience)。

### 3.1 仿真数据类型 (Data Simulated)
* **基础价格 (Mid-price)**: 依然使用 GBM 模拟市场宏观公允价值的演化。
* **订单簿状态 (LOB Depths)**: 仿真了 3 档买盘 (`Bid_1, Bid_2, Bid_3`)。
* **吃单逻辑 (Walking the Book)**: 废弃了人为定义的数学冲击公式 `g(v)`。智能体的卖出操作会直接“吃掉”买盘容量，产生物理滑点。若砸穿 3 档，将触发深度惩罚价格。
* **流动性弹性 (Resilience)**: 引入恢复速率 `kappa`。被吃掉的买盘会在随后的时间步中以 `kappa` 的速率逐渐恢复至基础容量 `base_vol`。

### 3.2 MDP 定义 (非马尔可夫特性)
* **State Space (状态空间)**: `[t_norm, q_norm, S_norm, Bid1_vol, Bid2_vol, Bid3_vol]`。维度：6。
* **核心难点 (Non-Markovian)**: 现在的市场状态不仅取决于当前，还取决于过去的动作（过去卖得太狠会导致当前 `Bid_vol` 枯竭）。这要求智能体必须具备“等一等”的微观结构操盘智慧 (Order Splitting)。

---

## 4. Evaluated Agent Architectures (评估的 AI 架构)
针对上述两个环境，我们对比了两种不同的深度强化学习架构：
1. **MLP (多层感知机 / 无记忆网络)**: 在 Env 1 (马尔可夫环境) 中表现优异，完美逼近 PDE 理论极值。但在 Env 2 中由于缺乏对流动性恢复的预判，表现受限。
2. **LSTM (长短期记忆网络 / 时序网络)**: 能够通过隐状态 (Hidden States) 记忆过去的订单簿消耗情况。在 Env 2 中，预期能展现出更接近真实交易员的“间歇性抛售” (Intermittent Execution) 策略。




你这个问题问得非常专业，直击高频交易和市场微观结构（Market Microstructure）的灵魂！从“真空公式”过渡到“限价单簿 (LOB)”，最难的一关就是参数校准 (Calibration)。如果在论文里，你的 LOB 参数是随便拍脑袋定的，答辩评委（特别是做量化的教授）一眼就能看穿。这三个核心超参数（tick_size, base_vol, kappa）在真实市场中都有明确的物理意义。我来帮你把这三者的经济学逻辑理顺，并给你一套“拿来就能写进论文”的设定依据。1. 最小报价单位 (tick_size)物理意义：交易所规定的价格变动最小刻度（比如 A 股是 0.01 元，美股通常是 0.01 美元，而加密货币可能是 0.1 甚至更小）。对 RL 的影响：它决定了你砸穿一档买盘时，价格会“掉”多少。tick_size 越大（比如 1.0），吃单的滑点惩罚就越恐怖，市场称为“大 tick 资产”；tick_size 越小（比如 0.01），价格曲线越平滑。设定建议：由于你的基础价格设为 $S_0 = 100$，你可以将 tick_size 设定为 0.05 到 0.1。这模拟了一只流动性不错的大盘股（滑点约为万分之五到千分之一）。2. 档位基础容量 (base_vol)物理意义：在没有大单冲击时，市场自然挂在买一、买二、买三上的订单数量。它代表了市场的静态深度 (Market Depth)。对 RL 的影响：它决定了你的 RL 智能体“一口气能吃个多大的胖子”。如果 base_vol = 0.2，而智能体想一秒钟卖 0.5，它就会瞬间砸穿两档半的买盘。高级洞见 (写进论文的加分项)：真实的 LOB 并不是平的！通常买一（Bid 1）的挂单量较少，买二、买三的挂单量会递增。如果你想把仿真做绝，可以在 reset 函数里这样初始化：self.bids_vol = np.array([0.5 * base_vol, 1.0 * base_vol, 1.5 * base_vol])。这叫“楔形订单簿”，极具学术价值！3. 流动性恢复速率 ($\kappa$ - kappa) —— 最核心的参数物理意义：当你砸出一个坑（吃掉了买盘）后，其他做市商（Market Makers）发现价格偏离了公允价值，跑过来重新挂单把坑填满的速度。这在学术界被称为市场弹性 (Market Resilience)，最经典的文献是 Obizhaeva & Wang (2013) 模型。如何科学地设定 $\kappa$？学术界通常用“半衰期 (Half-life)”来解释 $\kappa$。公式为：$\text{Half-life} = \frac{\ln(2)}{\kappa}$。如果你设定 $\kappa = 5.0$，半衰期大约是 0.14 个时间单位。这意味着只要停下不卖，短短 0.14 秒后，被你吃掉的买盘容量就会自动恢复 50%。这代表一个极其活跃、流动性极好的市场。如果你设定 $\kappa = 0.5$，半衰期就是 1.38。这代表一个流动性枯竭的垃圾股，砸出一个坑后半天都恢复不过来，RL 必须极其耐心地等待。为了让你和你的同伴对这三个参数如何相互作用产生直观的体感，我为你写了一个交互式的 LOB 弹性仿真器。你可以拖拽滑块，自己“砸一次盘”，看看订单簿是如何崩溃又如何被 $\kappa$ 修复的。




### 增补至 05_simulation_and_lob_design.md：

## 5. LOB Data Generation Mechanism (LOB 虚拟数据生成机制)

在本研究中，我们并未采用多变量 GBM 来独立生成各个订单簿档位的价格或容量。相反，我们构建了一个基于物理机制的**弹性订单簿模型 (Resilient LOB Model)**，该模型受到 Obizhaeva & Wang (2013) 的启发。

LOB 数据的生成遵循以下严格的微观结构动力学：

1.  **宏观锚定 (Macro Anchor)**:
    * 存在一个不可见的“公允基础价格 (Fair Mid-Price)” $S_t$，它遵循标准的几何布朗运动 (GBM)，代表市场对该资产基本面的宏观共识。

2.  **微观网格 (Micro Grid)**:
    * 限价单簿的各档价格 (Tick Prices) 并非随机游走，而是严格锚定在 $S_t$ 之下，以固定的最小报价单位 `tick_size` 呈现离散的阶梯分布。例如，买一价为 $S_t - \text{tick\_size}$，买二价为 $S_t - 2 \times \text{tick\_size}$。

3.  **确定性的订单薄深度 (Deterministic Depth Dynamics)**:
    * 订单簿的容量 (Volume) 不是随机生成的，而是遵循一种确定性的**“消耗-恢复 (Depletion and Recovery)”** 微分方程。
    * **消耗 (Depletion)**: 当智能体执行市价卖单时，订单簿容量会严格按照“价格优先”的原则被物理扣减 (Walking the book)。
    * **恢复 (Recovery - $\kappa$ Mechanism)**: 当流动性被消耗后，各档容量 $V_i(t)$ 会根据市场弹性 $\kappa$ 自动向其稳态最大容量 $V_{max, i}$ 恢复，其演化方程为：
      $$dV_i(t) = \kappa \cdot (V_{max, i} - V_i(t)) dt$$
    * 这种设计确保了市场具备“记忆性”，智能体必须学会等待流动性恢复，而不是盲目应对随机噪声。