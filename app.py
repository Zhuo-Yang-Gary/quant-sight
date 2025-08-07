import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

# 页面配置
st.set_page_config(page_title="QuantSight", layout="wide")
st.title("📈 QuantSight - 股票趋势预测与策略回测")

# 侧边栏输入区
st.sidebar.header("🛠 设置")
ticker = st.sidebar.text_input("输入股票代码（如AAPL, TSLA, BABA）", "TSLA")
start_date = st.sidebar.date_input("开始日期", datetime.today() - timedelta(days=365*1))
end_date = st.sidebar.date_input("结束日期", datetime.today())

# 限制 end_date 不能大于今天 - 1
today = datetime.today().date()
if end_date >= today:
    end_date = today - timedelta(days=1)

short_window = st.sidebar.number_input("短期均线窗口（天）", min_value=1, max_value=100, value=20)
long_window = st.sidebar.number_input("长期均线窗口（天）", min_value=1, max_value=300, value=50)
forecast_days = st.sidebar.slider("预测未来天数（线性回归）", 1, 15, 5)

# 展示开发者信息
with st.sidebar.expander("👤 关于作者"):
    st.markdown("""
    **Zhuo Yang**  
    Bachelor of Computing, Major in Software Development  
    University of Sydney | Jul 2023 – Mar 2026  
    📍 Wolli Creek, NSW  
    📧 graysonyang@outlook.com  
    🔗 [GitHub 主页](https://github.com/Zhuo-Yang-Gary)
    """)

# 尝试获取数据
with st.spinner("正在获取股票数据..."):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error("未能获取到有效数据，请尝试修改日期或股票代码。")
            st.stop()
    except Exception as e:
        st.error(f"获取股票数据时出错：{e}")
        st.stop()

# 计算均线
data["SMA_short"] = ta.sma(data["Close"], length=short_window)
data["SMA_long"] = ta.sma(data["Close"], length=long_window)

# 回测策略
data["Signal"] = 0
data["Signal"][short_window:] = np.where(
    data["SMA_short"][short_window:] > data["SMA_long"][short_window:], 1, -1
)
data["Position"] = data["Signal"].shift()

# 可视化价格 + 均线
st.subheader("📊 股价与移动平均线")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data["Close"], label="收盘价")
ax.plot(data["SMA_short"], label=f"短期均线 ({short_window}天)")
ax.plot(data["SMA_long"], label=f"长期均线 ({long_window}天)")
ax.legend()
st.pyplot(fig)

# 策略收益
st.subheader("💹 策略回测收益")
data["Returns"] = data["Close"].pct_change()
data["Strategy"] = data["Returns"] * data["Position"]
cumulative = (1 + data[["Returns", "Strategy"]]).cumprod()

st.line_chart(cumulative)

# 简单预测（线性回归）
st.subheader("🔮 未来价格预测（线性回归）")
df = data.dropna().copy()
df["Days"] = np.arange(len(df)).reshape(-1, 1)
model = LinearRegression().fit(df["Days"], df["Close"])
future_days = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
future_preds = model.predict(future_days)

future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
forecast_df = pd.Series(future_preds, index=future_dates)

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(data["Close"], label="历史收盘价")
ax2.plot(forecast_df, label="预测价格", linestyle="--")
ax2.legend()
st.pyplot(fig2)