import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# ——— 全局 CSS（Times New Roman + 间距） ———
st.markdown("""
    <style>
    body, h1, h2, h3, h4, p, select, button {
        font-family: 'Times New Roman', serif !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ——— 页面配置 ———
st.set_page_config(page_title="MSFT & NVDA Stock Forecast", layout="wide")

# ——— 标题 ———
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 2rem;'>MSFT & NVDA Future Price Forecast</h1>",
    unsafe_allow_html=True
)

# ——— 侧边栏 ———
with st.sidebar:
    st.header("Select Company")
    company = st.selectbox("Company", ["Microsoft (MSFT)", "NVIDIA (NVDA)"])
    st.header("Forecast Horizon")
    horizon_label = st.selectbox("Horizon", ["1 Month", "3 Months", "6 Months", "1 Year"])
    horizon_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
    forecast_days = horizon_map[horizon_label]
    run = st.button("Load & Forecast")

# ——— 作者卡片（右侧固定） ———
st.markdown("""
    <style>
    #author-card {
      position: fixed;
      top: 120px;
      right: 20px;
      width: 280px;
      background-color: #2e2f31;
      color: #ffffff;
      padding: 16px;
      border-radius: 12px;
      box-shadow: -2px 2px 8px rgba(0,0,0,0.4);
      z-index: 1000;
      font-family: 'Times New Roman', serif;
    }
    #author-card a { color: #61afef; text-decoration: none; }
    </style>
    <div id="author-card">
      <h4>About the Author</h4>
      <p><b>Zhuo Yang</b><br/>
      B.Sc. Computing, Software Development<br/>
      University of Sydney (2023–2026)</p>
      <p>Location: Wolli Creek, NSW<br/>
      +61 431 598 186<br/>
      <a href="mailto:gravsonvana@outlook.com">gravsonvana@outlook.com</a></p>
    </div>
""", unsafe_allow_html=True)

# ——— 数据加载 & 预测 ———
def load_data(ticker: str) -> pd.DataFrame:
    path_map = {
        "MSFT": "data/Microsoft_stock_data.csv",
        "NVDA": "data/nvda_stock_data.csv"
    }
    path = path_map.get(ticker)
    if path and os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df[["Date", "Close"]].dropna().sort_values("Date")
        df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
        return df
    return pd.DataFrame()

def train_and_forecast(df: pd.DataFrame, days: int):
    train_df = df.iloc[:-days].copy()
    test_df = df.iloc[-days:].copy()

    model = Prophet(daily_seasonality=True)
    model.fit(train_df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # 对测试集的预测及置信区间
    pred_test = (
        forecast.set_index("ds")
                .reindex(test_df["ds"])
                .rename(columns={
                    "yhat": "yhat_test",
                    "yhat_lower": "yhat_test_lower",
                    "yhat_upper": "yhat_test_upper"
                })
                .reset_index()
    )

    # 未来期预测及置信区间
    future_fc = forecast[
        forecast["ds"] > train_df["ds"].max()
    ][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    # 计算评估指标
    y_true = test_df["y"].values
    y_pred = pred_test["yhat_test"].values
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    return train_df, test_df, pred_test, future_fc, (mape, rmse, r2)

def plot_results(train, test, pred_test, future_fc, metrics, display_name, horizon_label):
    mape, rmse, r2 = metrics

    st.markdown(f"""
        <div style="margin-top:2rem; margin-bottom:1rem;">
          <h3>Model Evaluation on Last {horizon_label}</h3>
          <ul>
            <li><b>MAPE:</b> {mape:.2%}</li>
            <li><b>RMSE:</b> {rmse:.2f}</li>
            <li><b>R²:</b> {r2:.3f}</li>
          </ul>
        </div>
    """, unsafe_allow_html=True)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))

    # 画历史与测试实际
    ax.plot(train["ds"], train["y"], color="white", label="Train Actual")
    ax.plot(test["ds"], test["y"], color="#61afef", label="Test Actual")

    # 测试区间预测 & CI
    ax.fill_between(
        pred_test["ds"],
        pred_test["yhat_test_lower"],
        pred_test["yhat_test_upper"],
        color="orange", alpha=0.3, label="95% CI (Test)"
    )
    ax.plot(pred_test["ds"], pred_test["yhat_test"], "--", color="orange", label="Forecast on Test")

    # 未来区间预测 & CI
    ax.fill_between(
        future_fc["ds"],
        future_fc["yhat_lower"],
        future_fc["yhat_upper"],
        color="gray", alpha=0.2, label="95% CI (Future)"
    )
    ax.plot(future_fc["ds"], future_fc["yhat"], "--", color="lime", label="Future Forecast")

    ax.set_title(f"{display_name} Forecast – {horizon_label}", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price (USD)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#2e2f31", edgecolor="white", labelcolor="white")

    # 居中展示图表
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.pyplot(fig)

# ——— 主流程 ———
if run:
    ticker = "MSFT" if "Microsoft" in company else "NVDA"
    df = load_data(ticker)
    display_name = company

    if df.empty:
        st.error("No data available. Please check the CSV file.")
    else:
        train_df, test_df, pred_test, future_fc, metrics = train_and_forecast(df, forecast_days)
        plot_results(train_df, test_df, pred_test, future_fc, metrics, display_name, horizon_label)