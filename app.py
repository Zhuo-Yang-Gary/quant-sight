
import streamlit as st
import pandas as pd
import os
from datetime import timedelta
from prophet import Prophet
import matplotlib.pyplot as plt

# ----------------- UI Config -----------------
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>MSFT & NVDA Future Price Forecast</h1>", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("Select Company")
    company = st.selectbox("Company", ["Microsoft (MSFT)", "NVIDIA (NVDA)"])
    forecast_option = st.selectbox("Forecast Horizon", ["1 Month", "3 Months", "6 Months", "1 Year"])
    forecast_days_map = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365
    }
    forecast_days = forecast_days_map[forecast_option]
    run = st.button("Load & Forecast")

    st.markdown("---")
    st.markdown("""
    <div style='background-color:#2c2c2c; padding:15px; border-radius:10px; color:white;'>
    <h4>About the Author</h4>
    <p><strong>Zhuo Yang</strong><br/>
    B.Sc. Computing, Software Development<br/>
    University of Sydney (2023–2026)</p>
    <p>Location: Wolli Creek, NSW<br/>
    Phone: +61 431 598 186<br/>
    Email: <a href='mailto:gravsonvana@outlook.com'>gravsonvana@outlook.com</a></p>
    </div>
    """, unsafe_allow_html=True)

# ----------------- Helper -----------------
def load_csv(ticker):
    file_map = {
        "MSFT": "data/Microsoft_stock_data.csv",
        "NVDA": "data/nvda_stock_data.csv"
    }
    file = file_map.get(ticker.upper())
    if file and os.path.exists(file):
        df = pd.read_csv(file, parse_dates=["Date"])
        df = df[["Date", "Close"]].dropna().sort_values("Date")
        return df
    return None

def forecast_with_prophet(df, forecast_days):
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    forecast_df = forecast[["ds", "yhat"]].tail(forecast_days)
    return forecast_df

# ----------------- Main -----------------
if run:
    ticker = company.split("(")[-1].replace(")", "")
    df = load_csv(ticker)

    if df is not None:
        forecast_df = forecast_with_prophet(df, forecast_days)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecasted Close", color="orange", linestyle="--")
        ax.set_facecolor("#111")
        fig.patch.set_facecolor("#111")
        ax.tick_params(colors='white')
        ax.set_title(f"{company} Stock Price Forecast - {forecast_option}", color="white")
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Price (USD)", color="white")
        ax.legend()

        st.pyplot(fig)
    else:
        st.error("No data available. Please check the CSV file.")
