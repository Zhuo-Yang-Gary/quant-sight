import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ---------------- Sidebar: Stock Input -------------------
with st.sidebar:
    st.markdown("### Select Company")
    stock_option = st.selectbox(
        "Company",
        options=["Microsoft (MSFT)", "NVIDIA (NVDA)"]
    )
    ticker = stock_option.split("(")[-1].replace(")", "")

    st.markdown("### Forecast Horizon")
    forecast_horizon = st.selectbox(
        "Forecast Horizon",
        options=["1 Month", "3 Months", "6 Months", "1 Year"]
    )

    if forecast_horizon == "1 Month":
        forecast_days = 30
    elif forecast_horizon == "3 Months":
        forecast_days = 90
    elif forecast_horizon == "6 Months":
        forecast_days = 180
    elif forecast_horizon == "1 Year":
        forecast_days = 365

    run_forecast = st.button("Load & Forecast")

# ---------------- Author Card -------------------
with st.sidebar:
    st.markdown("---")
    with st.container():
        st.markdown(
            """
            <div style="background-color: #2c2c2c; padding: 15px; border-radius: 10px; color: white;">
                <h4 style="margin-bottom: 10px;">About the Author</h4>
                <p style="margin: 0;"><strong>Zhuo Yang</strong><br/>
                B.Sc. Computing, Software Development<br/>
                University of Sydney (2023–2026)</p>
                <p style="margin: 0;">Location: Wolli Creek, NSW</p>
                <p style="margin: 0;">Email: <a href="mailto:gravsonvana@outlook.com" style="color: lightblue;">gravsonvana@outlook.com</a></p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- Main Section -------------------
st.markdown("<h1 style='text-align: center;'>MSFT & NVDA Future Price Forecast</h1>", unsafe_allow_html=True)

def load_csv(ticker):
    file_map = {
        "MSFT": "Microsoft_stock_data.csv",
        "NVDA": "nvda_stock_data.csv"
    }
    file = file_map.get(ticker.upper())
    if file and os.path.exists(file):
        df = pd.read_csv(file, parse_dates=["Date"])
        df = df[["Date", "Close"]].dropna().sort_values("Date")
        return df
    return None

def forecast_stock(df, forecast_days):
    df = df.copy()
    df["Date_ordinal"] = df["Date"].map(pd.Timestamp.toordinal)
    model = LinearRegression()
    model.fit(df[["Date_ordinal"]], df["Close"])

    future_dates = [df["Date"].max() + timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_ordinals = [d.toordinal() for d in future_dates]
    predictions = model.predict(np.array(future_ordinals).reshape(-1, 1))

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Close": predictions
    })
    return df, future_df

if run_forecast:
    data = load_csv(ticker)
    if data is not None:
        historical, forecasted = forecast_stock(data, forecast_days)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(historical["Date"], historical["Close"], label="Historical Close", color="skyblue")
        ax.plot(forecasted["Date"], forecasted["Predicted Close"], label="Forecasted Close", linestyle="--", color="orange")
        ax.set_title(f"{ticker} Stock Price Forecast ({forecast_horizon})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("No data available. Please check the CSV file.")