import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="AI Stock Forecast Dashboard", layout="wide")
st.title("🤖 AI Sector Stock Trend Forecast (1M / 3M / 6M / 1Y)")

# ========== SELECTED STOCKS ==========
stock_dict = {
    "NVIDIA (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Alphabet (GOOGL)": "GOOGL",
    "Meta Platforms (META)": "META",
    "Amazon (AMZN)": "AMZN"
}

# ========== LAYOUT COLUMNS ==========
left, main, right = st.columns([1, 3, 1])

# ========== LEFT SIDEBAR ==========
with left:
    st.header("🧠 Select a Company")
    company_name = st.selectbox("Company", list(stock_dict.keys()))
    ticker = stock_dict[company_name]

# ========== DATE RANGE ==========
end_date = datetime.today().date()
start_date = end_date - timedelta(days=365 * 2)

# ========== FETCH DATA ==========
@st.cache_data
def fetch_stock_data(ticker):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.dropna(inplace=True)
        return data
    except:
        return pd.DataFrame()

data = fetch_stock_data(ticker)

# ========== MAIN DISPLAY ==========
with main:
    if data.empty:
        st.error("⚠️ Failed to retrieve stock data. Please check the ticker or network connection.")
    else:
        st.subheader(f"📈 Historical Closing Price: {ticker}")
        st.line_chart(data["Close"])

        # Forecasting
        df = data.copy()
        df["Days"] = np.arange(len(df)).reshape(-1, 1)
        X = df["Days"]
        y = df["Close"]

        model = LinearRegression().fit(X, y)

        future_periods = {
            "1 Month": 22,
            "3 Months": 66,
            "6 Months": 132,
            "1 Year": 264
        }

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data.index, data["Close"], label="Historical Price")

        for label, days in future_periods.items():
            future_x = np.arange(len(df), len(df) + days).reshape(-1, 1)
            future_y = model.predict(future_x)
            future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=days)
            ax.plot(future_dates, future_y, linestyle="--", label=f"{label} Forecast")

        ax.set_title(f"{company_name} Forecast (Linear Regression)")
        ax.legend()
        st.pyplot(fig)

# ========== RIGHT SIDEBAR (ABOUT) ==========
with right:
    st.header("👤 About the Author")
    st.markdown("""
    **Zhuo Yang**  
    Bachelor of Computing, Major in Software Development  
    University of Sydney | Jul 2023 – Mar 2026  

    📍 Wolli Creek, NSW  
    📧 graysonyang@outlook.com  
    📱 +61 431 598 186

    🔗 [GitHub Profile](https://github.com/Zhuo-Yang-Gary)

    ---
    This project demonstrates the ability to build interactive data-driven dashboards,  
    apply linear modeling for forecasting, and visualize real-world AI sector stock trends.
    """)