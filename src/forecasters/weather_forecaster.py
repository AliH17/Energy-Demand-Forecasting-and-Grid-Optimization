# src/forecasters/weather_forecaster.py

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.data_loader import load_weather_data
from xgboost                   import XGBRegressor
class WeatherForecaster:
    """
    Fast, hourly Holt–Winters on your station CSV for precipitation and avg_temp.
    """
    def __init__(
        self,
        station_csv: str,
        seasonal_periods: int = 24,
        trend: str = 'add',
        seasonal: str = 'add'
    ):
        self.station_csv      = station_csv
        self.seasonal_periods = seasonal_periods
        self.trend            = trend
        self.seasonal         = seasonal
        self.models           = {}
        self._fitted          = False

    def fit(self):
        """Load history and fit a Holt–Winters model for each target."""
        # limit to last year to keep fit super‐fast
        df = load_weather_data(self.station_csv).last('365D')
        for col in ['precipitation', 'avg_temp']:
            hist = df[col].astype(float)
            model = ExponentialSmoothing(
                hist,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method='estimated'
            )
            # suppress occasional overflow warnings
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fit_res = model.fit(optimized=True)
            self.models[col] = fit_res

        self._last_timestamp = df.index.max()
        self._fitted = True

    def predict(self, horizon: int = 24) -> pd.DataFrame:
        """
        Returns a DataFrame of shape (horizon, 2) with columns:
        ['precipitation','avg_temp'] for the next `horizon` hours.
        """
        if not self._fitted:
            self.fit()

        future_index = pd.date_range(
            start=self._last_timestamp + pd.Timedelta(hours=1),
            periods=horizon,
            freq='h'
        )
        out = pd.DataFrame(index=future_index)

        for col, fit_res in self.models.items():
            out[col] = fit_res.forecast(horizon).values

        return out
