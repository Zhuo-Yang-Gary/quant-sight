import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# ——— Page Configuration ———
st.set_page_config(page_title="MSFT & NVDA Stock Forecast", layout="wide")
st.markdown("<h1 style='text-align: center;'>MSFT & NVDA Future Price Forecast</h1>",
            unsafe_allow_html=True)

# ——— Sidebar ———
with st.sidebar:
    st.header("Select Company")
    company = st.selectbox("Company", ["Microsoft (MSFT)", "NVIDIA (NVDA)"])
    st.header("Forecast Horizon")
    horizon_label = st.selectbox("Horizon", ["1 Month", "3 Months", "6 Months", "1 Year"])
    horizon_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
    forecast_days = horizon_map[horizon_label]
    run = st.button("Load & Forecast")
    st.markdown("---")
    st.markdown(
        """
        <div style='background:#2e2f31; padding:16px; border-radius:10px; color:#fff; position:fixed; bottom:20px; width:260px;'>
          <h4>About the Author</h4>
          <p><b>Zhuo Yang</b><br/>
          B.Sc. Computing, Software Development<br/>
          University of Sydney (2023–2026)</p>
          <p>Wolli Creek, NSW<br/>
          +61 431 598 186<br/>
          <a style='color:#61afef;' href='mailto:gravsonvana@outlook.com'>gravsonvana@outlook.com</a></p>
        </div>
        """, unsafe_allow_html=True
    )

# ——— Helpers ———
def load_data(ticker: str) -> pd.DataFrame:
    path_map = {"MSFT": "data/Microsoft_stock_data.csv",
                "NVDA": "data/nvda_stock_data.csv"}
    path = path_map.get(ticker)
    if path and os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df[["Date", "Close"]].dropna().sort_values("Date")
        df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
        return df
    return pd.DataFrame()

def train_and_forecast(df: pd.DataFrame, days: int):
    # split train/test
    train_df = df.iloc[:-days]
    test_df = df.iloc[-days:]
    # fit Prophet
    m = Prophet(daily_seasonality=True)
    m.fit(train_df)
    # make future frame
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    # slice predictions
    test_fc = forecast.set_index("ds").loc[test_df["ds"]]
    future_fc = forecast[forecast["ds"] > train_df["ds"].max()]
    # compute metrics
    y_true = test_df["y"].values
    y_pred = test_fc["yhat"].values
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return train_df, test_df, future_fc, (mape, rmse, r2), forecast

def plot_results(train, test, future_fc, metrics, forecast_df, display_name, horizon_label):
    mape, rmse, r2 = metrics
    # display metrics
    st.markdown(f"""
    ### Model Evaluation on Last {horizon_label}
    - **MAPE:** {mape:.2%}  
    - **RMSE:** {rmse:.2f}  
    - **R²:** {r2:.3f}  
    """)
    # plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    # historical
    ax.plot(train["ds"], train["y"], color="white", label="Train Actual")
    ax.plot(test["ds"], test["y"], color="#61afef", label="Test Actual")
    # CI & test forecast
    test_fc = forecast_df.set_index("ds").loc[test["ds"]]
    ax.fill_between(test["ds"],
                    test_fc["yhat_lower"], test_fc["yhat_upper"],
                    color="orange", alpha=0.3, label="95% CI (Test)")
    ax.plot(test["ds"], test_fc["yhat"], "--", color="orange", label="Forecast on Test")
    # future forecast
    fc = future_fc
    ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                    color="gray", alpha=0.2, label="95% CI (Future)")
    ax.plot(fc["ds"], fc["yhat"], "--", color="lime", label="Future Forecast")
    ax.set_title(f"{display_name} Forecast – {horizon_label}", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price (USD)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#2e2f31", edgecolor="white", labelcolor="white")
    st.pyplot(fig)

# ——— Main Execution ———
if run:
    ticker = "MSFT" if "Microsoft" in company else "NVDA"
    df = load_data(ticker)
    display_name = company
    if df.empty:
        st.error("No data available. Please check the CSV file.")
    else:
        train_df, test_df, future_fc, metrics, forecast_df = train_and_forecast(df, forecast_days)
        plot_results(train_df, test_df, future_fc, metrics, forecast_df, display_name, horizon_label)