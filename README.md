# Energy-Demand-Forecasting-and-Grid-Optimization
Energy Demand Forecasting and Grid Optimization
This repository contains an end-to-end prototype for forecasting solar and load, and then using reinforcement learning to optimize battery dispatch to minimize grid imports. It includes:

    Data loaders for solar, load, and weather time series.

    Forecasting scripts (run_forecasters.py) that train XGBoost models on historical data and produce 24 h ahead forecasts.

    Gym environment (EnergyEnv) modeling a battery, prosumer load, and solar generation.

    RL training (train_rl.py) using Stable-Baselines3 DQN to learn a charge/discharge policy.

    Streamlit dashboard (app.py) to visualize forecasts and simulated dispatch.

Features

    Solar & Load Forecasting

        Feature engineering with Fourier terms, lags, rolling windows

        GPU-accelerated XGBoost models

        Train/test evaluation with MAE, RMSE, R²

    Battery Management Environment

        State = [normalized solar, normalized load, battery SoC]

        Actions = {hold, charge, discharge}

        Reward = –|net grid draw| – cycling penalty (with optional self-consumption incentive)

    Reinforcement Learning

        DQN agent trained on week-long episodes

        Evaluation callback and model checkpointing

        Sanity-check visualization of cumulative reward

    Interactive Dashboard

        Next-24 h forecasts of load & solar

        Simulated battery level & agent actions

Getting Started
1. Clone & Install

git clone https://github.com/yourusername/energy-grid-rl.git
cd energy-grid-rl
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

2. Prepare Data

Download and unzip:

    Solar generation from Kaggle: solar_power_generation_data.csv → data/solar/

    Load consumption from UCI: household_power_consumption.txt → data/load/

    Weather from NOAA GHCN: AE000041196.csv → data/renewables/

3. Run Forecasting

python -m scripts.run_forecasters
# → prints MAE, RMSE, R² for solar & load; saves forecasts_next24h.csv

4. Train RL Agent

python -m scripts.train_rl
# → trains DQN for 50k timesteps, checkpoints under models/, shows sanity-check plot

5. Launch Dashboard

streamlit run app.py

Use the sidebar to adjust cycle_penalty and toggle use_incentive, then click Simulate to view forecasts, battery dispatch, net grid draw, and rewards.
Interpretation of Metrics

    MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) gauge forecast accuracy in kWh or kW.

    R² > 0.8 indicates the model explains over 80 % of the variance in the test data—considered a strong fit for time-series applications.



        Net grid draw & cumulative reward plots

        Raw simulation table and reward-shaping controls
