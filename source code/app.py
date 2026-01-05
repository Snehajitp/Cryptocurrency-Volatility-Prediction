import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Crypto Volatility Predictor")

model = joblib.load("models/volatility_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Cryptocurrency Volatility Prediction")

uploaded_file = st.file_uploader("Upload Processed Crypto CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    features = [
        "open", "high", "low", "close",
        "volume", "market_cap",
        "daily_return", "ma_7", "ma_14",
        "liquidity_ratio", "high_low_spread"
    ]

    X = df[features]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    df["Predicted Volatility"] = predictions

    st.success("Prediction completed!")
    st.dataframe(df.tail(10))

    st.line_chart(df["Predicted Volatility"])
