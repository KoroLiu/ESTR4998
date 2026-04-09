
"""
Nonlinear Almgren-Chriss model with GBM + quadratic-variation penalty
--------------------------------------------------------------------
This script implements the discrete-time dynamic-programming scheme
described in Section 4 of the thesis draft.

Model:
    dq_t = -v_t dt
    dS_t = sigma * S_t dW_t
    cost_t = ((f(Q_t) + g(v_t)) * v_t + 0.5 * lam * sigma^2 * S_t^2 * q_t^2) dt
    terminal = 0.5 * alpha * S_T^2 * q_T^2

Objective:
    maximize E[ - sum_t cost_t - terminal ]

We use the quadratic-impact special case:
    f(Q) = gamma1 * Q + gamma2 * Q^2
    g(v) = eta1   * v + eta2   * v^2

Main outputs:
    - value function V[n, i, j]
    - optimal feedback policy v_star[n, i, j]
    - simulation utility for optimal and TWAP paths
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ModelParams:
    T: float = 1.0
    q0: float = 1.0
    S0: float = 100.0
    sigma: float = 0.2
    lam: float = 0.05
    alpha: float = 100.0
    gamma1: float = 0.02
    gamma2: float = 0.01
    eta1: float = 0.05
    eta2: float = 0.02


@dataclass
class GridParams:
    Nt: int = 40
    Nq: int = 60
    NS: int = 60
    Nv: int = 41
    S_min_mult: float = 0.4
    S_max_mult: float = 1.8
    gh_order: int = 5


def permanent_impact(Q: np.ndarray | float, p: ModelParams) -> np.ndarray | float:
    return p.gamma1 * Q + p.gamma2 * Q**2


def temporary_impact(v: np.ndarray | float, p: ModelParams) -> np.ndarray | float:
    return p.eta1 * v + p.eta2 * v**2


def build_grids(model: ModelParams, grid: GridParams):
    t = np.linspace(0.0, model.T, grid.Nt + 1)
    q = np.linspace(0.0, model.q0, grid.Nq + 1)
    s = np.linspace(grid.S_min_mult * model.S0, grid.S_max_mult * model.S0, grid.NS + 1)
    return t, q, s


def _interp_q_s(V_next: np.ndarray, q_grid: np.ndarray, s_grid: np.ndarray, qv: float, sv: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation of V_next(q, s) at one q value and many s values.
    V_next shape: (Nq+1, NS+1)
    Returns shape: same as sv
    """
    qv = float(np.clip(qv, q_grid[0], q_grid[-1]))
    sv = np.clip(sv, s_grid[0], s_grid[-1])

    iq = np.searchsorted(q_grid, qv, side="right") - 1
    iq = int(np.clip(iq, 0, len(q_grid) - 2))

    is_ = np.searchsorted(s_grid, sv, side="right") - 1
    is_ = np.clip(is_, 0, len(s_grid) - 2)

    q0, q1 = q_grid[iq], q_grid[iq + 1]
    s0, s1 = s_grid[is_], s_grid[is_ + 1]

    # Safe weights even at boundaries
    wq = 0.0 if q1 == q0 else (qv - q0) / (q1 - q0)
    ws = np.where(s1 == s0, 0.0, (sv - s0) / (s1 - s0))

    V00 = V_next[iq, is_]
    V01 = V_next[iq, is_ + 1]
    V10 = V_next[iq + 1, is_]
    V11 = V_next[iq + 1, is_ + 1]

    V_low = (1.0 - ws) * V00 + ws * V01
    V_high = (1.0 - ws) * V10 + ws * V11
    return (1.0 - wq) * V_low + wq * V_high


def solve_dp(model: ModelParams, grid: GridParams, verbose: bool = True):
    """
    Solve the discrete-time dynamic-programming approximation:
        V_n(q, S) = max_v E[ -running_cost*dt + V_{n+1}(q-vdt, S_next) ]
    """
    t_grid, q_grid, s_grid = build_grids(model, grid)
    dt = t_grid[1] - t_grid[0]

    # Gauss-Hermite quadrature for standard normal expectation
    gh_x, gh_w = np.polynomial.hermite.hermgauss(grid.gh_order)
    z = np.sqrt(2.0) * gh_x
    w = gh_w / np.sqrt(np.pi)

    V = np.zeros((grid.Nt + 1, grid.Nq + 1, grid.NS + 1))
    policy = np.zeros((grid.Nt, grid.Nq + 1, grid.NS + 1))

    # Terminal condition: V(T,q,S) = - alpha/2 * S^2 * q^2
    qq, ss = np.meshgrid(q_grid, s_grid, indexing="ij")
    V[-1] = -0.5 * model.alpha * ss**2 * qq**2

    for n in range(grid.Nt - 1, -1, -1):
        V_next = V[n + 1]

        if verbose:
            print(f"Backward step {n+1}/{grid.Nt}")

        for i, q_now in enumerate(q_grid):
            if q_now <= 1e-14:
                V[n, i, :] = 0.0
                policy[n, i, :] = 0.0
                continue

            # admissible controls keep q_next >= 0
            v_max = q_now / dt
            v_candidates = np.linspace(0.0, v_max, grid.Nv)

            Q_now = model.q0 - q_now
            perm = permanent_impact(Q_now, model)

            for j, s_now in enumerate(s_grid):
                best_val = -np.inf
                best_v = 0.0

                # exact GBM one-step transition
                s_next_nodes = s_now * np.exp((-0.5 * model.sigma**2) * dt + model.sigma * np.sqrt(dt) * z)

                for v in v_candidates:
                    q_next = q_now - v * dt
                    tmp = temporary_impact(v, model)
                    running = ((perm + tmp) * v + 0.5 * model.lam * model.sigma**2 * s_now**2 * q_now**2) * dt
                    cont_vals = _interp_q_s(V_next, q_grid, s_grid, q_next, s_next_nodes)
                    total = -running + np.sum(w * cont_vals)

                    if total > best_val:
                        best_val = total
                        best_v = v

                V[n, i, j] = best_val
                policy[n, i, j] = best_v

    return {
        "t_grid": t_grid,
        "q_grid": q_grid,
        "s_grid": s_grid,
        "V": V,
        "policy": policy,
        "dt": dt,
        "model": model,
        "grid": grid,
    }


def policy_at(solution: dict, n: int, q_now: float, s_now: float) -> float:
    q_grid = solution["q_grid"]
    s_grid = solution["s_grid"]
    pol = solution["policy"][n]

    q_now = float(np.clip(q_now, q_grid[0], q_grid[-1]))
    s_now = float(np.clip(s_now, s_grid[0], s_grid[-1]))

    iq = np.searchsorted(q_grid, q_now, side="right") - 1
    is_ = np.searchsorted(s_grid, s_now, side="right") - 1
    iq = int(np.clip(iq, 0, len(q_grid) - 2))
    is_ = int(np.clip(is_, 0, len(s_grid) - 2))

    q0, q1 = q_grid[iq], q_grid[iq + 1]
    s0, s1 = s_grid[is_], s_grid[is_ + 1]

    wq = 0.0 if q1 == q0 else (q_now - q0) / (q1 - q0)
    ws = 0.0 if s1 == s0 else (s_now - s0) / (s1 - s0)

    p00 = pol[iq, is_]
    p01 = pol[iq, is_ + 1]
    p10 = pol[iq + 1, is_]
    p11 = pol[iq + 1, is_ + 1]

    p_low = (1.0 - ws) * p00 + ws * p01
    p_high = (1.0 - ws) * p10 + ws * p11
    return max(0.0, float((1.0 - wq) * p_low + wq * p_high))


def simulate_path(solution: dict, seed: int = 0, strategy: str = "optimal"):
    """
    Simulate one path under either:
        strategy = 'optimal'  -> use solved feedback policy
        strategy = 'twap'     -> constant-speed liquidation
    """
    rng = np.random.default_rng(seed)

    model = solution["model"]
    t_grid = solution["t_grid"]
    dt = solution["dt"]
    Nt = len(t_grid) - 1

    q = np.zeros(Nt + 1)
    S = np.zeros(Nt + 1)
    v = np.zeros(Nt)
    cash = np.zeros(Nt + 1)
    Qcum = np.zeros(Nt + 1)

    q[0] = model.q0
    S[0] = model.S0

    for n in range(Nt):
        if strategy == "optimal":
            v[n] = min(policy_at(solution, n, q[n], S[n]), q[n] / dt)
        elif strategy == "twap":
            remaining_steps = Nt - n
            v[n] = q[n] / (remaining_steps * dt)
        else:
            raise ValueError("strategy must be 'optimal' or 'twap'")

        Qcum[n] = model.q0 - q[n]
        exec_price = S[n] - permanent_impact(Qcum[n], model) - temporary_impact(v[n], model)
        traded_cash = exec_price * v[n] * dt
        cash[n + 1] = cash[n] + traded_cash

        q[n + 1] = max(0.0, q[n] - v[n] * dt)

        z = rng.normal()
        S[n + 1] = S[n] * np.exp((-0.5 * model.sigma**2) * dt + model.sigma * np.sqrt(dt) * z)

    Qcum[-1] = model.q0 - q[-1]
    wealth = cash + q * S
    return {
        "t": t_grid,
        "q": q,
        "S": S,
        "v": v,
        "cash": cash,
        "wealth": wealth,
        "Qcum": Qcum,
    }


def evaluate_objective(path: dict, model: ModelParams, dt: float) -> float:
    """
    Realized objective on one path:
        -sum_t [ (f(Q_t)+g(v_t)) v_t + 0.5*lam*sigma^2*S_t^2*q_t^2 ] dt
        -0.5*alpha*S_T^2*q_T^2
    """
    q = path["q"][:-1]
    S = path["S"][:-1]
    v = path["v"]
    Qcum = path["Qcum"][:-1]

    running = ((permanent_impact(Qcum, model) + temporary_impact(v, model)) * v
               + 0.5 * model.lam * model.sigma**2 * S**2 * q**2)
    terminal = 0.5 * model.alpha * path["S"][-1]**2 * path["q"][-1]**2
    return -np.sum(running) * dt - terminal


def compare_strategies(solution: dict, n_paths: int = 200, seed: int = 123):
    model = solution["model"]
    dt = solution["dt"]
    vals_opt = []
    vals_twap = []
    for k in range(n_paths):
        p_opt = simulate_path(solution, seed=seed + k, strategy="optimal")
        p_twap = simulate_path(solution, seed=seed + k, strategy="twap")
        vals_opt.append(evaluate_objective(p_opt, model, dt))
        vals_twap.append(evaluate_objective(p_twap, model, dt))
    return {
        "optimal_mean": float(np.mean(vals_opt)),
        "optimal_std": float(np.std(vals_opt)),
        "twap_mean": float(np.mean(vals_twap)),
        "twap_std": float(np.std(vals_twap)),
        "optimal_all": np.array(vals_opt),
        "twap_all": np.array(vals_twap),
    }
