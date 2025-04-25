# ⚡ Energy Demand Forecasting & Grid Optimization

A proof-of-concept pipeline for 24 h ahead solar & load forecasting, and RL-based battery dispatch to minimize grid imports. Built with XGBoost, Gymnasium, Stable-Baselines3 DQN and Streamlit.

---


---

## Overview

This repository demonstrates an end-to-end workflow for:

1. **Solar & Load Forecasting** using GPU-accelerated XGBoost  
2. **Battery Management Environment** (Gymnasium)  
3. **Reinforcement Learning** (DQN) for charge/discharge policy  
4. **Interactive Dashboard** (Streamlit) to visualize forecasts, dispatch, and rewards  

By combining accurate short-term forecasts with RL, you can intelligently dispatch storage to smooth net grid draw and reduce costs.

---

## Features

- **Data Loaders**  
  - `load_solar_data`, `load_load_data`, `load_weather_data`  
- **Forecasting** (`scripts/run_forecasters.py`)  
  - Fourier features, lags, rolling-windows  
  - MAE/RMSE/R² evaluation  
  - Saves `forecasts_next24h.csv`  
- **Gym Environment** (`src/envs/energy_env.py`)  
  - State = [norm_solar, norm_load, batt_level]  
  - Actions = hold/charge/discharge  
  - Configurable cycling penalty & incentives  
- **RL Training** (`scripts/train_rl.py`)  
  - Stable-Baselines3 DQN + EvalCallback  
  - Checkpointing & policy sanity-check  
- **Dashboard** (`app.py`)  
  - Sidebar controls for penalty & incentive  
  - Plots: forecasts, battery SoC, net draw, rewards  

---
![image](https://github.com/user-attachments/assets/9558b1e4-a8bc-4880-8132-37c4ff0fbe4c)


---

## Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/energy-grid-rl.git
cd energy-grid-rl
python3 -m venv venv
source venv/bin/activate      # Windows: `venv\Scripts\activate`
pip install -r requirements.txt
2. Prepare Data

Download and place under data/:

    Solar generation (Kaggle) → data/solar/

    Household consumption (UCI) → data/load/

    Weather (NOAA GHCN: station AE000041196) → data/renewables/

3. Run Forecasting

python -m scripts.run_forecasters

Outputs MAE/RMSE/R² for solar & load and writes forecasts_next24h.csv.
4. Train RL Agent

python -m scripts.train_rl

Trains DQN for 50 k timesteps, saves best models under models/.
5. Launch Dashboard

streamlit run app.py

Use the sidebar to tweak cycle_penalty & use_incentive, then Simulate to see:

    Forecast curves

    Battery SoC & actions

    Net grid draw

    Cumulative reward
```

## Metrics

    MAE / RMSE measure absolute and squared forecast errors.

    R² > 0.8 indicates the model explains over 80 % of variance—a strong fit for time series.

## License

This project is MIT-licensed. See LICENSE for details.

