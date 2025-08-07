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

# ——— Author Info Banner at Top (Scrolls with page) ———
with st.container():
    st.markdown("""
    <div style='
        width: 100%;
        background-color: #2d2d2d;
        color: white;
        padding: 10px 25px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 14px;
    '>
        <strong>About the Author</strong> — Zhuo Yang, B.Sc. Computing (Software Development), University of Sydney (2023–2026). 
        Location: Wolli Creek, NSW · <a href="mailto:gravsonvana@outlook.com" style="color:#79b8ff;">gravsonvana@outlook.com</a> · +61 431 598 186
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>MSFT & NVDA Future Price Forecast</h1>", unsafe_allow_html=True)

# ——— Sidebar ———

with st.sidebar:
    st.title("Stock Forecast App")
    st.markdown("""
    **Instructions**:
    - Select a company to forecast.
    - Choose forecast horizon (e.g., 1 Year).
    - Click `Load & Forecast` to view prediction.
    """)

    st.header("Select Company")
    company = st.selectbox("Company", ["Microsoft (MSFT)", "NVIDIA (NVDA)"])
    st.header("Forecast Horizon")
    horizon_label = st.selectbox("Horizon", ["1 Month", "3 Months", "6 Months", "1 Year"])
    horizon_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
    forecast_days = horizon_map[horizon_label]
    run = st.button("Load & Forecast")

author_html = """
<style>
#author-card {
  position: fixed;
  top: 80px;        /* 下移到标题正下方 */
  right: 20px;
  width: 280px;
  background-color: #2e2f31;
  color: #ffffff;
  padding: 16px;
  border-radius: 12px;
  box-shadow: -2px 2px 8px rgba(0,0,0,0.4);
  z-index: 1000;
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
"""
st.markdown(author_html, unsafe_allow_html=True)

# ——— Helpers ———
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
    m = Prophet(daily_seasonality=True)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    pred_test = (
        forecast.set_index("ds")
                .reindex(test_df["ds"])
                .rename(columns={"yhat": "yhat_test",
                                 "yhat_lower": "yhat_test_lower",
                                 "yhat_upper": "yhat_test_upper"})
                .reset_index()
    )
    future_fc = forecast[forecast["ds"] > train_df["ds"].max()][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()
    y_true = test_df["y"].values
    y_pred = pred_test["yhat_test"].values
    import numpy as np
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    if len(y_true_clean) == 0:
        raise ValueError("All values in y_true or y_pred are NaN or inf.")
    import numpy as np
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    if len(y_true_clean) == 0:
        raise ValueError("All values in y_true or y_pred are NaN or inf.")
    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
    rmse = mean_squared_error(y_true_clean, y_pred_clean, squared=False)
    r2 = r2_score(y_true_clean, y_pred_clean)
    return train_df, test_df, pred_test, future_fc, (mape, rmse, r2)

def plot_results(train, test, pred_test, future_fc, metrics, display_name, horizon_label):
    mape, rmse, r2 = metrics
    st.markdown(f"""
    ### Model Evaluation on Last {horizon_label}
    - **MAPE:** {mape:.2%}  
    - **RMSE:** {rmse:.2f}  
    - **R²:** {r2:.3f}
    """)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train["ds"], train["y"], color="white", label="Train Actual")
    ax.plot(test["ds"], test["y"], color="#61afef", label="Test Actual")
    ax.fill_between(pred_test["ds"],
                    pred_test["yhat_test_lower"],
                    pred_test["yhat_test_upper"],
                    color="orange", alpha=0.3, label="95% CI (Test)")
    ax.plot(pred_test["ds"], pred_test["yhat_test"], "--", color="orange", label="Forecast on Test")
    ax.fill_between(future_fc["ds"],
                    future_fc["yhat_lower"],
                    future_fc["yhat_upper"],
                    color="gray", alpha=0.2, label="95% CI (Future)")
    ax.plot(future_fc["ds"], future_fc["yhat"], "--", color="lime", label="Future Forecast")
    ax.set_title(f"{display_name} Forecast – {horizon_label}", color="white")
    ax.set_xlabel("Date", color="white"); ax.set_ylabel("Price (USD)", color="white")
    ax.tick_params(colors="white"); ax.legend(facecolor="#2e2f31", edgecolor="white", labelcolor="white")
    st.pyplot(fig)

# ——— Main Execution ———
if run:
    ticker = "MSFT" if "Microsoft" in company else "NVDA"
    df = load_data(ticker)
    display_name = company
    if df.empty:
        st.error("No data available. Please check the CSV file.")
    else:
        train_df, test_df, pred_test, future_fc, metrics = train_and_forecast(df, forecast_days)
        plot_results(train_df, test_df, pred_test, future_fc, metrics, display_name, horizon_label)