#!/usr/bin/env python
import sys, os, time

# ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from src.envs.energy_env               import EnergyEnv
from src.forecasters.weather_forecaster import WeatherForecaster
from src.data_loader                   import (
    load_solar_data,
    load_load_data,
    load_weather_data,
)

def ts():
    return time.strftime("%H:%M:%S")

print(f"[{ts()}] Starting RL training script")

# 1) Fetch 24h weather forecasts
t0 = time.time()
print(f"[{ts()}] 1) Fetching 24h weather forecasts…")
wf   = WeatherForecaster('data/renewables/AE000041196.csv')
exog = wf.predict(horizon=24)
print(f"[{ts()}]    → done in {time.time() - t0:.1f}s")

# 2) Load one week of history
t1 = time.time()
print(f"[{ts()}] 2) Loading historical data for env creation…")
T = 168
solar_df = load_solar_data('data/solar/')
solar_df['total_yield']  = solar_df['daily_yield_p1'] + solar_df['daily_yield_p2']
solar_df['hourly_yield'] = solar_df['total_yield'].diff().fillna(solar_df['total_yield']).clip(lower=0)
solar_fc = solar_df['hourly_yield'].iloc[-T:]

load_df_ = load_load_data('data/load/household_power_consumption.txt')
load_fc  = load_df_['Global_active_power'].iloc[-T:]
print(f"[{ts()}]    → loaded {T} records in {time.time() - t1:.1f}s")

# 3) Build Gymnasium env with reward‐shaping
t2 = time.time()
print(f"[{ts()}] 3) Building Gym environment…")
env = EnergyEnv(
    solar_fc,
    load_fc,
    cycle_penalty=0.05,    # smaller cycling penalty
    use_incentive=True     # flip on incentive reward
)
print(f"[{ts()}]    → done in {time.time() - t2:.1f}s")

# 4) Hyperparam search + checkpointing via EvalCallback
t3 = time.time()
print(f"[{ts()}] 4) Training DQN w/ EvalCallback…")
eval_env = EnergyEnv(solar_fc, load_fc, cycle_penalty=0.05, use_incentive=True)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="models/",
    log_path="logs/",
    eval_freq=5_000,
    deterministic=True,
    render=False
)

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=10_000,
    batch_size=64,
    exploration_fraction=0.2,
    target_update_interval=500,
    train_freq=4,
    verbose=1,
)
model.learn(total_timesteps=50_000, callback=eval_callback)
print(f"[{ts()}]    → training done in {time.time() - t3:.1f}s")

# 5) Policy sanity-check
print(f"[{ts()}] 5) Sanity-checking policy on one episode…")
obs, _      = env.reset()
records     = []
for _ in range(T):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, info = env.step(action)
    records.append((obs.copy(), action, reward))
    if term or trunc:
        break

# turn into DataFrame
df = pd.DataFrame(records, columns=["obs","action","reward"])
df["cum_reward"] = df["reward"].cumsum()
plt.figure(figsize=(8,4))
plt.plot(df["cum_reward"], label="Cumulative Reward")
plt.xlabel("Timestep"); plt.ylabel("Cum. Reward")
plt.title("Policy Sanity-Check"); plt.legend()
plt.tight_layout()
plt.show()

# 6) Save final model
t5 = time.time()
print(f"[{ts()}] 6) Saving final model…")
os.makedirs("models", exist_ok=True)
model.save("models/energy_dqn_final")
print(f"[{ts()}]    → saved in {time.time() - t5:.1f}s")

print(f"[{ts()}] All done!")
