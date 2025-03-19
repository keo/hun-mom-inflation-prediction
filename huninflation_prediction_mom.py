"""
Hungarian M/M Inflation Prediction using Holt-Winters Method
This script fetches HICP data from Eurostat and predicts the next two month's inflation rate.
"""

import numpy as np
import pandas as pd
import eurostat
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fetch_eurostat_data():
    """Fetch and filter Hungarian HICP data from Eurostat."""
    try:
        df = eurostat.get_data_df('prc_hicp_mmor')
        if df.empty:
            raise ValueError("No data received from Eurostat")

        df_hungary = df[
            (df['geo\\TIME_PERIOD'] == 'HU') &
            (df['coicop'] == 'CP00') &
            (df['freq'] == 'M') &
            (df['unit'] == 'RCH_M')
        ]

        if df_hungary.empty:
            raise ValueError("No Hungarian data found after filtering")

        return df_hungary
    except Exception as e:
        raise RuntimeError(f"Error fetching data from Eurostat: {str(e)}") from e

def prepare_data(df_hungary):
    """Prepare and validate the data for analysis."""
    try:
        df_hungary = pd.melt(df_hungary,
                            id_vars=['freq', 'unit', 'coicop', 'geo\\TIME_PERIOD'],
                            var_name='Date',
                            value_name='Inflation')

        df_hungary['Date'] = pd.to_datetime(df_hungary['Date'], format='%Y-%m')
        df_hungary = df_hungary.sort_values(by='Date')
        df_hungary = df_hungary.dropna(subset=['Inflation'])
        df_hungary['Inflation'] = pd.to_numeric(df_hungary['Inflation'], errors='coerce')
        df_hungary = df_hungary.dropna(subset=['Inflation'])

        if df_hungary.empty:
            raise ValueError("No valid inflation data after cleaning")

        mean_inflation = df_hungary['Inflation'].mean()
        std_inflation = df_hungary['Inflation'].std()
        outliers = np.abs(df_hungary['Inflation'] - mean_inflation) > (3 * std_inflation)
        if outliers.any():
            print(f"Warning: {outliers.sum()} outliers detected in the data")

        df_hungary.set_index('Date', inplace=True)
        df_hungary = df_hungary[['Inflation']]
        df_hungary = df_hungary.sort_index()
        df_hungary = df_hungary.resample('M').last().ffill()

        if len(df_hungary) < 36:
            raise ValueError(
                "Insufficient data points after preparation. "
                f"Need at least 36, but got {len(df_hungary)}"
            )

        return df_hungary
    except Exception as e:
        raise RuntimeError(f"Error in data preparation: {str(e)}") from e

def calculate_mse(actual, predicted):
    """Calculate Mean Squared Error between actual and predicted values."""
    return np.mean((actual - predicted) ** 2)

def predict_inflation(df, periods=2):
    """
    Predict inflation using Holt-Winters method with fallback to moving average.
    
    Args:
        df: DataFrame containing inflation data
        periods: Number of periods to forecast (default: 2)
        
    Returns:
        dict: Contains forecast values, next months, and model statistics
    """
    required_months = 36
    if len(df) < required_months:
        msg = f"Insufficient data: need {required_months} months, but have {len(df)} months"
        raise ValueError(msg)

    # Add validation for maximum safe periods
    max_safe_periods = min(12, len(df)//3)  # Never more than 1 year, and 1/3 of history length
    if periods > max_safe_periods:
        raise ValueError(f"Maximum safe prediction period is {max_safe_periods} months")

    next_months = pd.date_range(start=df.index[-1], periods=periods+1, freq='M')[1:]

    try:
        model = ExponentialSmoothing(
            df['Inflation'],
            trend='add',
            seasonal='add',
            seasonal_periods=12,
            initialization_method='estimated',
            damped_trend=True
        )
        fit = model.fit(optimized=True)
        forecast = fit.forecast(periods)
        fitted_values = fit.fittedvalues
        mse = calculate_mse(df['Inflation'][-len(fitted_values):], fitted_values)

        return {
            'forecast': forecast,
            'next_months': next_months,
            'model_stats': {'aic': fit.aic, 'bic': fit.bic, 'mse': mse}
        }
    except (ValueError, RuntimeError) as e:
        print(f"Holt-Winters prediction failed: {str(e)}")
        print("Falling back to simple moving average prediction...")
        ma = df['Inflation'].rolling(window=12, min_periods=6).mean()
        if ma.empty or pd.isna(ma.iloc[-1]):
            raise ValueError("Unable to calculate moving average - insufficient data") from e

        forecast = pd.Series([ma.iloc[-1]] * periods, index=next_months)
        return {
            'forecast': forecast,
            'next_months': next_months,
            'model_stats': None
        }

def main():
    """Main function to run the inflation prediction process."""
    try:
        print("Fetching data from Eurostat...")
        df_hungary = fetch_eurostat_data()
        print(f"Shape of raw data: {df_hungary.shape}")

        print("\nPreparing data for analysis...")
        df_processed = prepare_data(df_hungary)
        df = df_processed.tail(36)

        print("\nRecent actual values:")
        print(df['Inflation'].tail(6).to_string())

        for month, forecast_value in zip(result['next_months'], result['forecast']):
            print(f"Estimated inflation for {month.strftime('%Y-%m')}: {forecast_value:.2f}%")

        print("\nGenerating predictions...")
        result = predict_inflation(df, periods=3)

        if result['model_stats']:
            print("\nModel Statistics:")
            print(f"AIC: {result['model_stats']['aic']:.2f}")
            print(f"BIC: {result['model_stats']['bic']:.2f}")
            print(f"MSE: {result['model_stats']['mse']:.2f}")

    except (RuntimeError, ValueError) as e:
        print(f"Error: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
