import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LiquidationEnvLOB(gym.Env):
    """
    基于限价单簿 (Limit Order Book) 机制的清算环境。
    引入了离散订单簿层级、吃单滑点以及流动性的弹性恢复机制 (Resilience)。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 q0=1.0,         # 初始库存
                 S0=100.0,       # 基础价格 (Mid-price)
                 T=1.0,          # 总时间
                 N=100,          # 时间步数
                 sigma=0.2,      # 基础价格波动率
                 lam=0.5,        # 风险厌恶系数
                 kappa=5.0,      # 流动性恢复速率 (弹性)
                 tick_size=0.1,  # 每档订单簿的价差
                 base_vol=0.2,   # 每档订单簿的基础容量
                 alpha=100.0,    # 终局未清仓惩罚
                 reward_scale=1e-3):
        super(LiquidationEnvLOB, self).__init__()
        
        self.q0 = q0
        self.S0 = S0
        self.T = T
        self.N = N
        self.dt = T / N
        self.sigma = sigma
        self.lam = lam
        self.kappa = kappa
        self.tick_size = tick_size
        self.base_vol = base_vol
        self.alpha = alpha
        self.reward_scale = reward_scale
        
        # 动作空间：[-1, 1] 映射到卖出速率
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 观察空间增加：[时间, 库存, 基础价格, 买一量, 买二量, 买三量]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
            high=np.array([1.0, 1.0, np.inf, np.inf, np.inf, np.inf]), 
            dtype=np.float32
        )

    # 检查括号里有没有 z_path=None
    def reset(self, seed=None, options=None, z_path=None): 
        super().reset(seed=seed)
        self.current_step = 0
        self.q = self.q0
        self.S = self.S0
        self.z_path = z_path # <-- 必须有这一行把它存下来
        
        # 楔形订单簿初始化
        self.bids_vol = np.array([
            1.0 * self.base_vol, 
            2.0 * self.base_vol, 
            3.0 * self.base_vol
        ], dtype=np.float32)
        self.max_bids_vol = np.copy(self.bids_vol)
        
        return self._get_obs(), {}

    def _get_obs(self):
        t_norm = self.current_step / self.N
        q_norm = self.q / self.q0
        S_norm = self.S / self.S0
        # 归一化订单簿容量
        vols_norm = self.bids_vol / self.base_vol 
        obs = np.array([t_norm, q_norm, S_norm, vols_norm[0], vols_norm[1], vols_norm[2]], dtype=np.float32)
        return obs

    def step(self, action):
        # 1. 基础价格演化 (GBM) - 锁死随机数！
        if self.z_path is not None and self.current_step < len(self.z_path):
            Z = self.z_path[self.current_step]
        else:
            Z = np.random.normal(0, 1)
            
        self.S = self.S * np.exp((-0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)
        
        # 2. 订单簿流动性自动恢复 (注意这里用 max_bids_vol)
        recovery = self.kappa * (self.max_bids_vol - self.bids_vol) * self.dt
        self.bids_vol = np.clip(self.bids_vol + recovery, 0.0, self.max_bids_vol)
        
        # 3. 动作映射 (防止超卖)
        time_left = max(self.T - self.current_step * self.dt, self.dt)
        v_avg = self.q / time_left
        v_max = 3.0 * v_avg if v_avg > 0 else 0.0
        v_t = np.clip((action[0] + 1.0) / 2.0 * v_max, 0.0, v_max)
        v_t = min(v_t, self.q / self.dt)
        
        trade_qty = v_t * self.dt
        remaining_qty = trade_qty
        total_revenue = 0.0
        
        # 4. 模拟吃单 (Walking the Book)
        prices = [self.S - self.tick_size, self.S - 2*self.tick_size, self.S - 3*self.tick_size]
        
        for i in range(3):
            if remaining_qty <= 0:
                break
            # 能吃掉的量是订单簿这档的剩余量和自己想卖的量取最小值
            execute_qty = min(remaining_qty, self.bids_vol[i])
            total_revenue += execute_qty * prices[i]
            self.bids_vol[i] -= execute_qty
            remaining_qty -= execute_qty
            
        # 如果把 3 档全砸穿了还有剩，剩下的只能以极低的价格成交 (惩罚性滑点)
        if remaining_qty > 0:
            penalty_price = self.S - 10 * self.tick_size
            total_revenue += remaining_qty * penalty_price
            
        # 计算成交均价 VWAP (如果没有卖出，设为当前 S)
        vwap = total_revenue / trade_qty if trade_qty > 0 else self.S
        
        # 真实执行成本 (完美无冲击执行应该是以 S 成交)
        exec_cost = (self.S * trade_qty) - total_revenue
        
        # 5. 计算奖励
        risk_penalty = 0.5 * self.lam * (self.sigma ** 2) * (self.S ** 2) * (self.q ** 2) * self.dt
        step_reward = -(exec_cost + risk_penalty)
        
        # 6. 状态更新
        self.q -= trade_qty
        self.current_step += 1
        
        terminated = bool(self.current_step >= self.N)
        if terminated:
            terminal_penalty = 0.5 * self.alpha * (self.S ** 2) * (self.q ** 2)
            step_reward -= terminal_penalty
            
        scaled_reward = step_reward * self.reward_scale

        info = {
            "inventory": self.q,
            "price": self.S,
            "execution_rate": v_t,
            "vwap": vwap,
            "bid1_vol": self.bids_vol[0]
        }
        
        return self._get_obs(), float(scaled_reward), terminated, False, info