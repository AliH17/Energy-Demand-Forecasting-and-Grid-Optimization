#!/usr/bin/env python
import sys, os, time
import pandas as pd
import numpy as np
from math import sqrt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost                   import XGBRegressor
from sklearn.metrics           import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.forecasters.weather_forecaster import WeatherForecaster
from src.data_loader            import (
    load_solar_data,
    load_load_data,
    load_weather_data
)

# ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def ts():
    return time.strftime("%H:%M:%S")

start = time.time()
print(f"[{ts()}] 🚀 Starting run_forecasts.py")


# ──────────────────────────────────────────────────────────────────────────────
# 1) SOLAR: load & feature-engineer exactly like the Jupyter notebook
# ──────────────────────────────────────────────────────────────────────────────
solar_df = load_solar_data('data/solar/').copy()

# 1a) true hourly generation from DAILY_YIELD
# diff gives Wh, so convert to kWh and floor at zero
solar_df['hy1'] = solar_df['daily_yield_p1'].diff().fillna(0)
solar_df['hy2'] = solar_df['daily_yield_p2'].diff().fillna(0)
solar_df['yield'] = ((solar_df['hy1'] + solar_df['hy2']).clip(lower=0)) 


# 1b) Time-of-day & annual Fourier
solar_df['hour']  = solar_df.index.hour
solar_df['doy']   = solar_df.index.dayofyear
solar_df['sin_h'] = np.sin(2 * np.pi * solar_df['hour'] / 24)
solar_df['cos_h'] = np.cos(2 * np.pi * solar_df['hour'] / 24)
solar_df['sin_d'] = np.sin(2 * np.pi * solar_df['doy']  / 365)
solar_df['cos_d'] = np.cos(2 * np.pi * solar_df['doy']  / 365)

# 1c) Lags & rolling-window stats
solar_df['lag_1h']   = solar_df['yield'].shift(1)
solar_df['lag_24h']  = solar_df['yield'].shift(24)
solar_df['rmean_3h'] = solar_df['yield'].rolling(3).mean()
solar_df['rstd_3h']  = solar_df['yield'].rolling(3).std()
solar_df['rmean_24h']= solar_df['yield'].rolling(24).mean()

# 1d) drop any rows with NaNs from shifts/rolls
mask = (
    solar_df[['avg_irradiation','avg_ambient_temp','avg_module_temp',
              'sin_h','cos_h','sin_d','cos_d',
              'lag_1h','lag_24h','rmean_3h','rstd_3h','rmean_24h']]
    .notna().all(axis=1)
)
solar_df = solar_df.loc[mask]

print(
    f" Solar history: {solar_df.index.min()} → {solar_df.index.max()}, "
    f"{len(solar_df)} samples"
)

# 1e) Train/test split (80/20)
split = int(len(solar_df) * 0.8)
print(f"  → train: {split} hrs,  test: {len(solar_df)-split} hrs")

X_s = solar_df[[
    'avg_irradiation','avg_ambient_temp','avg_module_temp',
    'sin_h','cos_h','sin_d','cos_d',
    'lag_1h','lag_24h','rmean_3h','rstd_3h','rmean_24h'
]]
y_s = solar_df['yield']
X_s_train, X_s_test = X_s.iloc[:split], X_s.iloc[split:]
y_s_train, y_s_test = y_s.iloc[:split], y_s.iloc[split:]


# ──────────────────────────────────────────────────────────────────────────────
# 2) SOLAR: fit XGB & evaluate exactly like the notebook
# ──────────────────────────────────────────────────────────────────────────────
solar_model = XGBRegressor(
    tree_method='hist',
    device='cuda',
    random_state=42,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.6,
    colsample_bytree=0.6,
    verbosity=0
)
t0 = time.time()
solar_model.fit(X_s_train, y_s_train)
print(f"[{ts()}] 1) Solar model trained in {time.time()-t0:.1f}s")

# full-day metrics
y_hat_s = solar_model.predict(X_s_test)
mae_s   = mean_absolute_error(y_s_test, y_hat_s)
rmse_s  = sqrt(mean_squared_error(y_s_test, y_hat_s))
r2_s    = r2_score(y_s_test, y_hat_s)
print(f"[{ts()}]    Solar test  — MAE: {mae_s:.2f} kWh   RMSE: {rmse_s:.2f} kWh   R²: {r2_s:.3f}")

# daytime-only metrics
day_mask    = X_s_test['avg_irradiation'] > 0
mae_day     = mean_absolute_error(y_s_test[day_mask],    y_hat_s[day_mask])
rmse_day    = sqrt(mean_squared_error(y_s_test[day_mask], y_hat_s[day_mask]))
r2_day      = r2_score(y_s_test[day_mask],                y_hat_s[day_mask])
print(f"[{ts()}]    Solar daytime — MAE: {mae_day:.2f} kWh   RMSE: {rmse_day:.2f} kWh   R²: {r2_day:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# 3) LOAD: load & feature-engineer
# ──────────────────────────────────────────────────────────────────────────────
load_df = load_load_data('data/load/household_power_consumption.txt')
wt      = load_weather_data('data/renewables/AE000041196.csv')
df_l    = load_df.join(wt[['precipitation','avg_temp']], how='inner').dropna()

df_l['hour']       = df_l.index.hour
df_l['dow']        = df_l.index.dayofweek
df_l['is_weekend'] = (df_l['dow'] >= 5).astype(int)
df_l['sin_hour']   = np.sin(2*np.pi * df_l['hour']    / 24)
df_l['cos_hour']   = np.cos(2*np.pi * df_l['hour']    / 24)
df_l['sin_dow']    = np.sin(2*np.pi * df_l['dow']     / 7)
df_l['cos_dow']    = np.cos(2*np.pi * df_l['dow']     / 7)

df_l['lag_1h']       = df_l['Global_active_power'].shift(1)
df_l['lag_24h']      = df_l['Global_active_power'].shift(24)
df_l['roll_mean_3h'] = df_l['Global_active_power'].rolling(3).mean()
df_l['roll_std_3h']  = df_l['Global_active_power'].rolling(3).std()
df_l['roll_mean_24h']= df_l['Global_active_power'].rolling(24).mean()

df_l.dropna(inplace=True)

print(
    f" Load history: {df_l.index.min()} → {df_l.index.max()}, "
    f"{len(df_l)} samples"
)

split_l = int(len(df_l) * 0.8)
print(f"  → train: {split_l} hrs,  test: {len(df_l)-split_l} hrs")

X_l = df_l[[
    'precipitation','avg_temp',
    'sin_hour','cos_hour','sin_dow','cos_dow','is_weekend',
    'lag_1h','lag_24h','roll_mean_3h','roll_std_3h','roll_mean_24h'
]]
y_l = df_l['Global_active_power']
X_l_train, X_l_test = X_l.iloc[:split_l], X_l.iloc[split_l:]
y_l_train, y_l_test = y_l.iloc[:split_l], y_l.iloc[split_l:]


# ──────────────────────────────────────────────────────────────────────────────
# 4) LOAD: fit & report
# ──────────────────────────────────────────────────────────────────────────────
load_params = {
    'tree_method':       'hist',
    'device':            'cuda',
    'random_state':      42,
    'max_depth':         8,
    'learning_rate':     0.07267561528460804,
    'n_estimators':      738,
    'subsample':         0.8073318609454947,
    'colsample_bytree':  0.9720067339243328,
    'gamma':             0.3520806542477195,
    'min_child_weight':  2
}
load_model = XGBRegressor(**load_params)
t1 = time.time()
load_model.fit(X_l_train, y_l_train)
print(f"[{ts()}] 2) Load model trained in {time.time()-t1:.1f}s")

yhat_l = load_model.predict(X_l_test)
mae_l  = mean_absolute_error(y_l_test, yhat_l)
rmse_l = sqrt(mean_squared_error(y_l_test, yhat_l))
r2_l   = r2_score(y_l_test, yhat_l)
print(
    f"[{ts()}]    Load test — MAE: {mae_l:.2f} kW    "
    f"RMSE: {rmse_l:.2f} kW    R²: {r2_l:.3f}"
)


# ──────────────────────────────────────────────────────────────────────────────
# 5) WEATHER → exog for next 24h & reorder
# ──────────────────────────────────────────────────────────────────────────────
t2 = time.time()
wf24      = WeatherForecaster('data/renewables/AE000041196.csv')
exog_24h  = wf24.predict(horizon=24)
print(f"[{ts()}] 3) Weather forecast done in {time.time()-t2:.1f}s")

# 5a) SOLAR exog (Holt–Winters + same 12 cols)
def hw(s, h=24):
    return ExponentialSmoothing(
        s, trend='add', seasonal='add', seasonal_periods=24
    ).fit(optimized=True).forecast(h)

# ─── 5a) SOLAR exog ───
# instead of pure sine→max, peg your proxy around the historical mean ± amplitude
# replace the entire clear-sky block with this:
irrad_fc = hw(solar_df['avg_irradiation'], 24)   # already in same units (kWh? or W/m²)
amb_fc   = hw(solar_df['avg_ambient_temp'],   24)
mod_fc   = hw(solar_df['avg_module_temp'],    24)

solar_exog = pd.DataFrame({
    'avg_irradiation':  irrad_fc,
    'avg_ambient_temp': amb_fc,
    'avg_module_temp':  mod_fc
}, index=irrad_fc.index)


# then build all the other features as before…
solar_exog['sin_h']   = np.sin(2*np.pi * solar_exog.index.hour      / 24)
solar_exog['cos_h']   = np.cos(2*np.pi * solar_exog.index.hour      / 24)
solar_exog['sin_d']   = np.sin(2*np.pi * solar_exog.index.dayofyear / 365)
solar_exog['cos_d']   = np.cos(2*np.pi * solar_exog.index.dayofyear / 365)
solar_exog['lag_1h']  = solar_df['yield'].iloc[-1]
solar_exog['lag_24h'] = solar_df['yield'].iloc[-24]
l3  = solar_df['yield'].iloc[-3:]
l24 = solar_df['yield'].iloc[-24:]
solar_exog['rmean_3h']  = l3.mean()
solar_exog['rstd_3h']   = l3.std()
solar_exog['rmean_24h'] = l24.mean()

solar_exog_24h = solar_exog[[
    'avg_irradiation','avg_ambient_temp','avg_module_temp',
    'sin_h','cos_h','sin_d','cos_d',
    'lag_1h','lag_24h','rmean_3h','rstd_3h','rmean_24h'
]]





# 5b) LOAD exog (same 12 cols)
last_l   = df_l['Global_active_power'].iloc[-1]
last_l24 = df_l['Global_active_power'].iloc[-24]
load_exog = pd.DataFrame({
    'precipitation': exog_24h['precipitation'],
    'avg_temp':      exog_24h['avg_temp']
}, index=exog_24h.index)

load_exog['sin_hour']     = np.sin(2*np.pi * load_exog.index.hour      / 24)
load_exog['cos_hour']     = np.cos(2*np.pi * load_exog.index.hour      / 24)
load_exog['sin_dow']      = np.sin(2*np.pi * load_exog.index.dayofweek / 7)
load_exog['cos_dow']      = np.cos(2*np.pi * load_exog.index.dayofweek / 7)
load_exog['is_weekend']   = (load_exog.index.dayofweek >= 5).astype(int)
load_exog['lag_1h']       = last_l
load_exog['lag_24h']      = last_l24
l3_  = df_l['Global_active_power'].iloc[-3:]
l24_ = df_l['Global_active_power'].iloc[-24:]
load_exog['roll_mean_3h']  = l3_.mean()
load_exog['roll_std_3h']   = l3_.std()
load_exog['roll_mean_24h'] = l24_.mean()

load_exog_24h = load_exog[[
    'precipitation','avg_temp',
    'sin_hour','cos_hour','sin_dow','cos_dow','is_weekend',
    'lag_1h','lag_24h','roll_mean_3h','roll_std_3h','roll_mean_24h'
]]


# ──────────────────────────────────────────────────────────────────────────────
# 6) PREDICT & SAVE
# ──────────────────────────────────────────────────────────────────────────────
t3 = time.time()
y_s = solar_model.predict(solar_exog_24h)
y_s = np.clip(y_s, 0, None)
y_l     = load_model.predict(load_exog_24h)
print(f"[{ts()}] 4) Forecasts generated in {time.time()-t3:.1f}s")



out = pd.DataFrame({
    'datetime':  exog_24h.index,
    'solar_kWh': y_s,
    'load_kW':   y_l
})
out.to_csv('forecasts_next24h.csv', index=False)

print(f"[{ts()}]  All done in {time.time()-start:.1f}s")

# ──────────────────────────────────────────────────────────────────────────────
# 7) RL POLICY → generate 24 h action plan
# ──────────────────────────────────────────────────────────────────────────────
from stable_baselines3 import DQN
from src.envs.energy_env import EnergyEnv

# load your final DQN model
rl_model = DQN.load("models/energy_dqn_final")

# wrap our forecasts into a Gym env
sol_series  = pd.Series(y_s, index=exog_24h.index)
load_series = pd.Series(y_l, index=exog_24h.index)
action_env = EnergyEnv(
    sol_series,
    load_series,
    cycle_penalty=0.05,    # match whatever you used during training
    use_incentive=False
)

# step through the env to collect actions
obs, _     = action_env.reset()
actions    = []
for _ in range(len(sol_series)):
    action, _ = rl_model.predict(obs, deterministic=True)
    obs, _, done, _, _ = action_env.step(action)
    actions.append(action)
    if done:
        break

# save to CSV
action_df = pd.DataFrame({
    "datetime": exog_24h.index,
    "action":   actions
})
action_df.to_csv("action_plan_next24h.csv", index=False)

print(f"[{ts()}] Action plan saved to action_plan_next24h.csv")
