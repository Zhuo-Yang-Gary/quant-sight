# QuantSight - 股票趋势预测与策略回测平台（Streamlit版本）

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------------
# 页面设置
# -------------------------------
st.set_page_config(page_title="QuantSight - 股票趋势预测", layout="wide")
st.title("📈 QuantSight - 股票趋势预测与策略回测")

# -------------------------------
# 用户输入部分
# -------------------------------
with st.sidebar:
    st.header("🛠️ 设置")
    ticker = st.text_input("输入股票代码（如AAPL, TSLA, BABA）", value="AAPL")
    start_date = st.date_input("开始日期", value=date.today() - timedelta(days=365))
    end_date = st.date_input("结束日期", value=date.today())
    short_ma = st.number_input("短期均线窗口（天）", min_value=1, value=20)
    long_ma = st.number_input("长期均线窗口（天）", min_value=1, value=50)
    predict_days = st.slider("预测未来天数（线性回归）", 1, 15, 5)

# -------------------------------
# 获取数据
# -------------------------------
def fetch_stock(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        df = df.dropna()
        return df
    except:
        return None

data = fetch_stock(ticker, start_date, end_date)

if data is None or data.empty:
    st.error("未能获取到有效数据，请检查股票代码或网络连接。")
    st.stop()

# -------------------------------
# 技术指标计算
# -------------------------------
data['SMA_short'] = data['Close'].rolling(window=short_ma).mean()
data['SMA_long'] = data['Close'].rolling(window=long_ma).mean()
data['Signal'] = 0

# 策略信号生成（简单均线交叉）
data.loc[data['SMA_short'] > data['SMA_long'], 'Signal'] = 1

data['Position'] = data['Signal'].diff()

# -------------------------------
# 图表展示
# -------------------------------
st.subheader(f"📊 股票价格与均线 ({ticker})")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data['Close'], label='收盘价', color='black')
ax.plot(data.index, data['SMA_short'], label=f'{short_ma}日均线', linestyle='--')
ax.plot(data.index, data['SMA_long'], label=f'{long_ma}日均线', linestyle='--')
ax.set_title(f"{ticker} 股价走势与均线")
ax.legend()
st.pyplot(fig)

# -------------------------------
# 策略信号展示
# -------------------------------
st.subheader("📍 策略信号点（红：买入，绿：卖出）")
buy = data[data['Position'] == 1]
sell = data[data['Position'] == -1]
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(data.index, data['Close'], label='收盘价', color='gray')
ax2.plot(buy.index, buy['Close'], '^', markersize=10, color='red', label='买入信号')
ax2.plot(sell.index, sell['Close'], 'v', markersize=10, color='green', label='卖出信号')
ax2.legend()
ax2.set_title("交易信号图")
st.pyplot(fig2)

# -------------------------------
# 简单线性回归预测未来N天价格
# -------------------------------
st.subheader("🔮 股票价格预测（线性回归）")
last_days = 30  # 用过去多少天数据来拟合趋势
train_data = data['Close'][-last_days:]
X = np.arange(len(train_data)).reshape(-1, 1)
y = train_data.values
model = LinearRegression().fit(X, y)
future_X = np.arange(len(train_data), len(train_data) + predict_days).reshape(-1, 1)
preds = model.predict(future_X)

# 显示预测图
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(data.index[-last_days:], y, label='历史数据')
future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=predict_days)
ax3.plot(future_dates, preds, label='预测价格', linestyle='--', color='orange')
ax3.set_title("未来价格预测（线性回归）")
ax3.legend()
st.pyplot(fig3)

st.success("✅ 数据加载与分析完成。您可以更换股票代码查看不同资产。")