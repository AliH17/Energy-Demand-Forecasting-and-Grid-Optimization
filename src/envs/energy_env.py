import numpy as np
import gymnasium as gym
from gymnasium import spaces

class EnergyEnv(gym.Env):
    """
    Battery‐management env with tunable reward shaping.

    State = [norm_solar, norm_load, batt_level]
    Action = Discrete([0=hold,1=charge,2=discharge])

    reward = base_reward
           - cycle_penalty * |delta|
           + incentive * (solar + discharge - load)
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        solar_fc,
        load_fc,
        capacity: float = 10.0,
        max_rate: float = 1.0,
        cycle_penalty: float = 0.1,
        use_incentive: bool = False,
    ):
        super().__init__()
        raw_solar = solar_fc.values.flatten()
        raw_load  = load_fc.values.flatten()
        # normalize
        self.solar = raw_solar / raw_solar.max()
        self.load  = raw_load  / raw_load.max()

        # battery in per-unit
        self.cap  = 1.0
        self.rate = max_rate / raw_load.max()

        # reward-shaping knobs
        self.cycle_penalty = cycle_penalty
        self.use_incentive = use_incentive

        self.T = len(self.solar)
        self.observation_space = spaces.Box(0.0, 1.0, (3,), dtype=np.float32)
        self.action_space      = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t    = 0
        self.batt = 0.5
        return np.array([self.solar[0], self.load[0], self.batt], dtype=np.float32), {}
    
    def step(self, action):
        a = int(np.array(action).item())
        delta = {0: 0.0, 1: +self.rate, 2: -self.rate}[a]

        # penalize invalid charge/discharge
        invalid_discharge = (a == 2 and self.batt <= 0.0)
        invalid_charge    = (a == 1 and self.batt >= self.cap)
        if invalid_discharge or invalid_charge:
            delta = 0.0

        # apply battery change
        self.batt = np.clip(self.batt + delta, 0.0, self.cap)

        # net grid draw
        net = self.load[self.t] - self.solar[self.t] - delta

        # base reward
        reward = -abs(net)
        reward -= self.cycle_penalty * abs(delta)

        if self.use_incentive:
            reward = (self.solar[self.t] + max(delta, 0.0)) - self.load[self.t]

        if invalid_discharge:
            reward -= 0.2
        if invalid_charge:
            reward -= 0.1

        self.t += 1
        terminated = self.t >= self.T
        truncated  = False

        if not terminated:
            obs = np.array([self.solar[self.t], self.load[self.t], self.batt], dtype=np.float32)
        else:
            obs = np.zeros(3, dtype=np.float32)

        return obs, reward, terminated, truncated, {}

    def render(self, mode="human"):
        # optional: print or log batt/net over time
        pass
