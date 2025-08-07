import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="MSFT & NVDA Stock Forecast", layout="wide")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("### Select Company")
    company = st.selectbox("Company", ["Microsoft (MSFT)", "NVIDIA (NVDA)"])

    st.markdown("### Forecast Horizon")
    forecast_option = st.selectbox("Forecast Horizon", ["1 Month", "3 Months", "6 Months", "1 Year"])

    load_button = st.button("Load & Forecast")

    st.markdown("---")
    with st.container():
        st.markdown(
            """
            <div style="background-color:#2e2f31; padding: 16px; border-radius: 12px;">
            <h4 style="color:white;">About the Author</h4>
            <p style="color:white;">
            <b>Zhuo Yang</b><br>
            B.Sc. Computing, Software Development<br>
            University of Sydney (2023–2026)<br>
            <br>
            Location: Wolli Creek, NSW<br>
            Phone: +61 431 598 186<br>
            Email: <a style='color: #61afef;' href='mailto:gravsonvana@outlook.com'>gravsonvana@outlook.com</a>
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------ Main Title ------------------
st.markdown("<h1 style='text-align: center;'>MSFT & NVDA Future Price Forecast</h1>", unsafe_allow_html=True)

# ------------------ Utility Functions ------------------

def load_csv(ticker):
    file_map = {
        "MSFT": "Microsoft_stock_data.csv",
        "NVDA": "nvda_stock_data.csv"
    }
    file = file_map.get(ticker.upper())
    file_path = os.path.join("data", file)
    if file and os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=["Date"])
        df = df[["Date", "Close"]].dropna().sort_values("Date")
        return df
    return None

def forecast_stock(df, forecast_days):
    df = df.copy()
    df["Date_ordinal"] = df["Date"].map(pd.Timestamp.toordinal)

    model = LinearRegression()
    model.fit(df[["Date_ordinal"]], df["Close"])

    last_date = df["Date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_ordinals = [d.toordinal() for d in future_dates]
    predictions = model.predict(np.array(future_ordinals).reshape(-1, 1))

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Close": predictions
    })
    return future_df

def plot_forecast(future_df, display_name, forecast_option):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.plot(future_df["Date"], future_df["Predicted Close"], linestyle='--', color='orange', label="Forecasted Close")
    ax.set_title(f"{display_name} Stock Price Forecast - {forecast_option}", color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Price (USD)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor="#0e1117", edgecolor="white", labelcolor='white')
    st.pyplot(fig)

# ------------------ Main Logic ------------------

if load_button:
    ticker = "MSFT" if "Microsoft" in company else "NVDA"
    display_name = company
    df = load_csv(ticker)

    if df is None or df.empty:
        st.error("No data available. Please check the CSV file.")
    else:
        forecast_days = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }.get(forecast_option, 30)

        future_df = forecast_stock(df, forecast_days)
        plot_forecast(future_df, display_name, forecast_option)