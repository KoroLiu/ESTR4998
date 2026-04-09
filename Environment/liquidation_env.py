import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LiquidationEnv(gym.Env):
    """
    基于 Almgren-Chriss 模型扩展的强化学习清算环境。
    包含 GBM 价格动态、非线性价格冲击以及二次变差风险惩罚。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 q0=1.0,         # 初始库存量
                 S0=100.0,       # 初始价格
                 T=1.0,          # 总时间范围
                 N=100,          # 时间步数 (离散化)
                 sigma=0.2,      # 波动率
                 gamma1=0.01,    # 线性永久冲击系数
                 gamma2=0.005,   # 二次永久冲击系数
                 eta1=0.01,      # 线性临时冲击系数
                 eta2=0.005,     # 二次临时冲击系数
                 lam=0.5,        # 风险厌恶系数 (lambda)
                 alpha=100.0,    # 终局未清仓惩罚系数
                 reward_scale=1e-3 # 奖励缩放因子，帮助 PPO 收敛
                 ):
        super(LiquidationEnv, self).__init__()
        
        # 保存参数
        self.q0 = q0
        self.S0 = S0
        self.T = T
        self.N = N
        self.dt = T / N
        self.sigma = sigma
        
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.eta1 = eta1
        self.eta2 = eta2
        self.lam = lam
        self.alpha = alpha
        self.reward_scale = reward_scale
        
        # 动作空间：设为 [-1, 1]，在 step 中会映射为实际卖出速率 v_t
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 观察空间：[当前时间步比例, 剩余库存比例, 归一化价格]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]), 
            high=np.array([1.0, 1.0, np.inf]), 
            dtype=np.float32
        )
        
        # 初始化状态变量
        self.current_step = 0
        self.q = self.q0
        self.S = self.S0

    def reset(self, seed=None, options=None, z_path=None): # <-- 增加 z_path 参数
        super().reset(seed=seed)
        self.current_step = 0
        self.q = self.q0
        self.S = self.S0
        self.z_path = z_path # <-- 保存这条路径
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # 返回归一化的状态
        t_normalized = self.current_step / self.N
        q_normalized = self.q / self.q0
        S_normalized = self.S / self.S0
        return np.array([t_normalized, q_normalized, S_normalized], dtype=np.float32)

    def step(self, action):
        # 1. 动作映射: 将 [-1, 1] 映射到 [0, v_max]
        # v_max 设为能在剩余时间内刚好卖完的平均速率的 2 倍，给予探索空间
        time_left = self.T - (self.current_step * self.dt)
        if time_left <= 0: time_left = self.dt
        v_avg = self.q / time_left
        v_max = 2.0 * v_avg if v_avg > 0 else 0.0
        
        # action 从 [-1, 1] -> [0, 1] -> [0, v_max]
        v_t = np.clip((action[0] + 1.0) / 2.0 * v_max, 0.0, v_max)
        
        # 确保不会超卖 (v_t * dt <= q)
        v_t = min(v_t, self.q / self.dt)
        
        # 2. 计算累积已卖出量 Q_t
        Q_t = self.q0 - self.q  #
        
        # 3. 计算冲击成本 (Impacts)
        # 永久冲击 f(Q_t) = \gamma_1 Q_t + \gamma_2 Q_t^2
        f_Q = self.gamma1 * Q_t + self.gamma2 * (Q_t ** 2)
        
        # 临时冲击 g(v_t) = \eta_1 v_t + \eta_2 v_t^2
        g_v = self.eta1 * v_t + self.eta2 * (v_t ** 2)
        
        # 4. 计算单步代价与奖励 (对应方程 47 里的瞬时成本)
        exec_cost = (f_Q + g_v) * v_t * self.dt
        risk_penalty = 0.5 * self.lam * (self.sigma ** 2) * (self.S ** 2) * (self.q ** 2) * self.dt
        
        step_reward = -(exec_cost + risk_penalty)
        
        self.q -= v_t * self.dt
        
        # GBM 价格更新: 优先使用传入的 z_path，否则随机生成
        if self.z_path is not None and self.current_step < len(self.z_path):
            Z = self.z_path[self.current_step]
        else:
            Z = np.random.normal(0, 1)
            
        self.S = self.S * np.exp((-0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)
        
        # 必须先用完 Z，再增加 step
        self.current_step += 1
        
        # 6. 检查是否终止及计算终局惩罚
        terminated = bool(self.current_step >= self.N)
        truncated = False
        
        if terminated:
            # 终局惩罚 (方程 41)
            terminal_penalty = 0.5 * self.alpha * (self.S ** 2) * (self.q ** 2)
            step_reward -= terminal_penalty
            
        # 奖励缩放
        scaled_reward = step_reward * self.reward_scale

        info = {
            "inventory": self.q,
            "price": self.S,
            "execution_rate": v_t,
            "execution_price": self.S - f_Q - g_v #
        }
        
        return self._get_obs(), float(scaled_reward), terminated, truncated, info