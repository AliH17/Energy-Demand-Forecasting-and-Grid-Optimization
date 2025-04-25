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

        Net grid draw & cumulative reward plots

        Raw simulation table and reward-shaping controls
