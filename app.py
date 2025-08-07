import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import time

# 页面基本配置
st.set_page_config(page_title="MSFT & NVDA Forecast", layout="wide")
st.title("MSFT & NVDA Future Price Forecast")

# ——— 侧边栏：公司与预测时长选择 ———
company = st.sidebar.selectbox(
    "Select Company",
    ["Microsoft (MSFT)", "NVIDIA (NVDA)"]
)
ticker = "MSFT" if company.startswith("Microsoft") else "NVDA"

forecast_options = {
    "1 Month": 22,
    "3 Months": 66,
    "6 Months": 132,
    "1 Year": 264
}
forecast_label = st.sidebar.selectbox("Forecast Horizon", list(forecast_options.keys()))
forecast_days = forecast_options[forecast_label]

# ——— 侧边栏：触发按钮 ———
if st.sidebar.button("Load & Forecast", use_container_width=True):

    # 进度条：开始 → 下载 → 处理 → 完成
    progress = st.progress(0, text="Initializing...")
    time.sleep(0.1)
    progress.progress(20, text="Downloading historical data...")

    # 下载过去三年数据
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=3*365)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_date, end=end_date)

    progress.progress(60, text="Processing data...")
    time.sleep(0.1)

    # 检查数据有效性
    if df.empty:
        st.error(f"No data found for {ticker}.")
        progress.empty()
    else:
        # 线性回归预测
        df = df.reset_index()
        df["Days"] = np.arange(len(df)).reshape(-1, 1)
        model = LinearRegression().fit(df[["Days"]], df["Close"])
        future_x = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
        future_y = model.predict(future_x)
        future_dates = pd.date_range(df["Date"].iloc[-1] + timedelta(days=1), periods=forecast_days)

        progress.progress(100, text="Forecast complete")
        time.sleep(0.2)
        progress.empty()

        # 可视化历史价格与预测
        st.subheader(f"{company} Historical & {forecast_label} Forecast")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["Date"], df["Close"], label="Historical")
        ax.plot(future_dates, future_y, linestyle="--", label="Forecast")
        ax.set_title(f"{ticker} Price Forecast")
        ax.legend()
        st.pyplot(fig)

# ——— 固定右上：作者介绍卡片 ———
author_card = """
<div style="
    position:fixed; top:20px; right:20px;
    width:240px; max-height:300px; overflow:auto;
    background-color:#2c2c2c; color:#ffffff;
    padding:15px; border-radius:12px;
    box-shadow:0 4px 12px rgba(0,0,0,0.3);
    z-index:100;
">
  <h4 style="margin:0 0 10px 0;">About the Author</h4>
  <p style="margin:0;"><strong>Zhuo Yang</strong></p>
  <p style="margin:0;">B.Sc. Computing, Software Development</p>
  <p style="margin:0;">University of Sydney (2023–2026)</p>
  <hr style="border:0.5px solid #444; margin:10px 0;"/>
  <p style="margin:0;">Location: Wolli Creek, NSW</p>
  <p style="margin:0;">Email: <a href="mailto:graysonyang@outlook.com" style="color:#9cf;">graysonyang@outlook.com</a></p>
  <p style="margin:0;"><a href="https://github.com/Zhuo-Yang-Gary" target="_blank" style="color:#9cf;">GitHub Profile</a></p>
</div>
"""
st.markdown(author_card, unsafe_allow_html=True)