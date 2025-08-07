import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import re
import time

# 页面配置
st.set_page_config(page_title="AI Stock Forecast Dashboard", layout="wide")
st.title("AI Stock Forecast Dashboard")

# ——— 左侧：输入股票代码与按钮 ———
ticker_input = st.sidebar.text_input("Ticker (1–4 letters)", max_chars=4)
ticker = ticker_input.strip().upper()
if ticker and not re.fullmatch(r"[A-Z]{1,4}", ticker):
    st.sidebar.error("Invalid format: Please enter 1–4 letters (A–Z only).")
    st.stop()

forecast_options = {
    "1 Month": 22,
    "3 Months": 66,
    "6 Months": 132,
    "1 Year": 264
}
forecast_label = st.sidebar.selectbox("Forecast Horizon", list(forecast_options.keys()))
forecast_days = forecast_options[forecast_label]

if st.sidebar.button("Load & Forecast", use_container_width=True):
    if not ticker:
        st.sidebar.error("Please input a ticker code.")
        st.stop()

    # ——— 进度条显示下载与计算过程 ———
    progress = st.progress(0, text="Starting...")
    time.sleep(0.2)
    progress.progress(20, text="Downloading data...")

    today = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start="2020-01-01", end=today)

    progress.progress(60, text="Processing data...")
    time.sleep(0.2)

    if df.empty:
        st.error(f"No data found for '{ticker}'.")
        progress.empty()
        st.stop()

    df.reset_index(inplace=True)
    df["Days"] = np.arange(len(df)).reshape(-1, 1)
    model = LinearRegression().fit(df[["Days"]], df["Close"])
    future_x = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    future_y = model.predict(future_x)
    future_dates = pd.date_range(df["Date"].iloc[-1] + timedelta(days=1), periods=forecast_days)

    progress.progress(100, text="Complete!")
    time.sleep(0.3)
    progress.empty()

    # ——— 图表展示 ———
    st.subheader(f"{ticker} Historical & {forecast_label} Forecast")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Close"], label="Historical")
    ax.plot(future_dates, future_y, linestyle="--", label="Forecast")
    ax.set_title(f"{ticker} Forecast")
    ax.legend()
    st.pyplot(fig)

# ——— 右侧：固定作者卡片 ———
author_card = """
<div style='position:fixed; right:30px; top:100px; width:320px;
            background-color:#2c2c2c; color:white; padding:20px;
            border-radius:12px; box-shadow:0px 4px 12px rgba(0,0,0,0.3);'>
  <h3>About the Author</h3>
  <p><strong>Zhuo Yang</strong></p>
  <p>B.Sc. Computing, Software Development</p>
  <p>University of Sydney (2023–2026)</p>
  <hr style='border:0.5px solid #444;'/>
  <p>Wolli Creek, NSW</p>
  <p>Email: <a href='mailto:graysonyang@outlook.com' style='color:#9cf;'>graysonyang@outlook.com</a></p>
  <p><a href='https://github.com/Zhuo-Yang-Gary' target='_blank' style='color:#9cf;'>GitHub Profile</a></p>
</div>
"""
st.markdown(author_card, unsafe_allow_html=True)