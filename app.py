import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import DQN

from src.envs.energy_env import EnergyEnv
from src.forecasters.weather_forecaster import WeatherForecaster
from src.data_loader import load_solar_data, load_load_data

# ──────────────────────────────────────────────────────────────────────────────
st.title("🏭 Energy + Battery RL Dashboard")
# ──────────────────────────────────────────────────────────────────────────────

# 1) Load 24h forecasts
@st.cache_data
def load_forecasts(path="forecasts_next24h.csv"):
    df = pd.read_csv(path, parse_dates=["datetime"], dayfirst=True)
    df.set_index("datetime", inplace=True)
    return df

fc = load_forecasts()
st.subheader("🔮 Next-24h Forecasts")
st.line_chart(fc[["solar_kWh","load_kW"]])

# 2) Load our trained policy
@st.cache_resource
def load_policy(path="models/energy_dqn_final.zip"):
    return DQN.load(path)

model = load_policy()

# 3) Build energy env and roll out policy
@st.cache_data
def simulate(fc, cycle_penalty, use_incentive):
    # get last T-hour history for env (you could also replay exactly the fc period)
    T = len(fc)
    # build env
    env = EnergyEnv(
        solar_fc   = pd.Series(fc["solar_kWh"].values, index=fc.index),
        load_fc    = pd.Series(fc["load_kW"].values,  index=fc.index),
        cycle_penalty=cycle_penalty,
        use_incentive=use_incentive
    )
    obs, _ = env.reset()
    records = []
    for _ in range(T):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, term, trunc, _ = env.step(action)
        records.append({
            "action": action,
            "battery": obs[2],
            "reward": reward,
            "net_grid": env.load[env.t-1] - env.solar[env.t-1] - ({0:0,1:env.rate,2:-env.rate}[action])
        })
        if term: break
    return pd.DataFrame(records, index=fc.index[:len(records)])

# allow user to tweak reward‐shaping knobs
st.sidebar.header("⚙️ Reward-shaping")
cycle_penalty  = st.sidebar.slider("cycle_penalty", 0.0, 1.0, 0.05, 0.01)
use_incentive  = st.sidebar.checkbox("use_incentive", True)

sim = simulate(fc, cycle_penalty, use_incentive)

# 4) Plots
st.subheader("🔋 Battery level & Actions")
fig, ax = plt.subplots(2,1, figsize=(8,4), sharex=True)
sim["battery"].plot(ax=ax[0], title="Battery Level (0–1pu)")
ax[0].set_ylabel("Batt pu")
sim["action"].plot(ax=ax[1], drawstyle="steps-post", title="Action (0=hold,1=charge,2=discharge)")
ax[1].set_ylabel("Action")
st.pyplot(fig)

st.subheader("⚡ Net grid draw & Reward")
fig, ax = plt.subplots(2,1, figsize=(8,4), sharex=True)
sim["net_grid"].plot(ax=ax[0], title="Net Grid Draw (kW)")
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_ylabel("kW")
(sim["reward"].cumsum()).plot(ax=ax[1], title="Cumulative Reward")
ax[1].set_ylabel("Reward")
st.pyplot(fig)

# 5) Data tables
with st.expander("► Show raw simulation table"):
    st.dataframe(sim)

st.markdown("> 👆 Use the sidebar to adjust reward-shaping and see how the policy changes.")
