# src/data_loader.py

import pandas as pd
from glob import glob

def load_load_data(path: str) -> pd.DataFrame:
    """Hourly DataFrame from UCI household power (.txt)."""
    df = pd.read_csv(
        path,
        sep=';',
        usecols=[
            'Date','Time','Global_active_power','Global_reactive_power',
            'Voltage','Global_intensity','Sub_metering_1',
            'Sub_metering_2','Sub_metering_3'
        ],
        na_values=['?']
    )
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        dayfirst=True
    )
    df = df.set_index('datetime').drop(columns=['Date','Time'])
    df = (
        df
        .apply(pd.to_numeric, errors='coerce')
        .resample('h').mean()
        .ffill()
    )
    return df


# src/data_loader.py

import os
from glob import glob
import pandas as pd

# src/data_loader.py

import pandas as pd
import os
from glob import glob

def load_solar_data(folder: str) -> pd.DataFrame:
    """
    Load & merge the 4 solar plant files in `folder` into an hourly DataFrame:
      - Plant_1_Generation_Data.csv    → DAILY_YIELD, TOTAL_YIELD
      - Plant_1_Weather_Sensor_Data.csv → IRRADIATION, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE
      - Plant_2_Generation_Data.csv
      - Plant_2_Weather_Sensor_Data.csv

    Returns a DataFrame with columns:
      daily_yield_p1, total_yield_p1,
      daily_yield_p2, total_yield_p2,
      avg_irradiation, avg_ambient_temp, avg_module_temp
    indexed by a DatetimeIndex at hourly frequency.
    """
    gen_paths    = sorted(glob(os.path.join(folder, "Plant_*_Generation_Data.*")))
    sensor_paths = sorted(glob(os.path.join(folder, "Plant_*_Weather_Sensor_Data.*")))

    plants = []
    for (g_path, s_path) in zip(gen_paths, sensor_paths):
        # --- generation data ---
        # support both .csv and Excel
        if g_path.lower().endswith(('.xls', '.xlsx')):
            g = pd.read_excel(g_path, parse_dates=['DATE_TIME'], index_col='DATE_TIME')
        else:
            g = pd.read_csv( g_path, parse_dates=['DATE_TIME'], index_col='DATE_TIME')

        g = (
            g[['DAILY_YIELD','TOTAL_YIELD']]
             .rename(columns={
                 'DAILY_YIELD': 'daily_yield',
                 'TOTAL_YIELD': 'total_yield'
             })
             .apply(pd.to_numeric, errors='coerce')
             # make hourly, forward-fill any gaps
             .resample('h').mean().ffill()
        )

        # --- sensor data ---
        if s_path.lower().endswith(('.xls', '.xlsx')):
            s = pd.read_excel(s_path, parse_dates=['DATE_TIME'], index_col='DATE_TIME')
        else:
            s = pd.read_csv( s_path, parse_dates=['DATE_TIME'], index_col='DATE_TIME')

        s = (
            s[['IRRADIATION','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']]
             .rename(columns={
                 'IRRADIATION': 'irradiation',
                 'AMBIENT_TEMPERATURE': 'ambient_temp',
                 'MODULE_TEMPERATURE': 'module_temp'
             })
             .apply(pd.to_numeric, errors='coerce')
             .resample('h').mean().ffill()
        )

        # join generation + sensors
        df_plant = g.join(s, how='inner')
        # drop any duplicate timestamps
        df_plant = df_plant[~df_plant.index.duplicated(keep='first')]

        # suffix with plant id (1 or 2)
        pid = os.path.basename(g_path).split('_')[1]  # "1" or "2"
        df_plant = df_plant.add_suffix(f"_p{pid}")
        plants.append(df_plant)

    # concat the two plants side-by-side
    df = pd.concat(plants, axis=1)

    # build the averaged sensor columns
    df['avg_irradiation']  = (df['irradiation_p1']  + df['irradiation_p2'])  / 2
    df['avg_ambient_temp'] = (df['ambient_temp_p1'] + df['ambient_temp_p2']) / 2
    df['avg_module_temp']  = (df['module_temp_p1']  + df['module_temp_p2'])  / 2

    # finally, return exactly the 7 columns you need
    return df[[
        'daily_yield_p1', 'total_yield_p1',
        'daily_yield_p2', 'total_yield_p2',
        'avg_irradiation','avg_ambient_temp','avg_module_temp'
    ]]




def load_weather_data(path: str) -> pd.DataFrame:
    """
    Hourly weather features from NOAA GHCN-Daily CSV.
    Expects columns: DATE, PRCP, TMAX, TMIN, TAVG.
    """
    df = pd.read_csv(
        path,
        usecols=['DATE','PRCP','TMAX','TMIN','TAVG'],
        na_values=['']
    )
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
    df = (
        df
        .set_index('DATE')
        .rename(columns={
            'PRCP':'precipitation',
            'TMAX':'max_temp',
            'TMIN':'min_temp',
            'TAVG':'avg_temp'
        })
        .apply(pd.to_numeric, errors='coerce')
        .resample('h').ffill().bfill()
    )
    return df
