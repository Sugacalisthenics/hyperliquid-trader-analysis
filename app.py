import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Primetrade AI Dashboard", layout="wide")

st.title("📈 Primetrade: Trader Performance vs Market Sentiment")
st.markdown("An interactive dashboard analyzing how Fear/Greed sentiment impacts trader behavior and profitability on Hyperliquid.")

# --- 1. DATA LOADING (ABSOLUTE PATHS FIXED) ---
@st.cache_data
def load_data():
    path_trades = r"C:\Users\Lenovo\Desktop\Primetrade_Assignment\historical_data.csv"
    path_sent = r"C:\Users\Lenovo\Desktop\Primetrade_Assignment\fear_greed_index.csv"
    
    df_trades = pd.read_csv(path_trades)
    df_sentiment = pd.read_csv(path_sent)
    
    # Bulletproof Date Parsing
    df_trades['date'] = pd.to_datetime(df_trades['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.strftime('%Y-%m-%d')
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.strftime('%Y-%m-%d')
    
    df_merged = pd.merge(df_trades, df_sentiment, on='date', how='inner')
    
    daily_metrics = df_merged.groupby(['date', 'Account']).agg(
        daily_PnL=('Closed PnL', 'sum'),
        total_trades=('Closed PnL', 'count'),
        avg_trade_size=('Size USD', 'mean')
    ).reset_index()
    
    daily_metrics = pd.merge(daily_metrics, df_sentiment[['date', 'classification', 'value']], on='date', how='left')
    return daily_metrics

df = load_data()

# --- 2. KPI METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Analyzed Trade Days", len(df))
col2.metric("Total Profit (Greed Days)", f"${df[df['classification'] == 'Greed']['daily_PnL'].sum():,.0f}")
col3.metric("Total Profit (Fear Days)", f"${df[df['classification'] == 'Fear']['daily_PnL'].sum():,.0f}")

st.divider()

# --- 3. VISUALIZATIONS ---
st.subheader("📊 Behavioral Shifts: Fear vs Greed")
colA, colB = st.columns(2)

with colA:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='classification', y='total_trades', palette="viridis", ax=ax)
    ax.set_title("Average Trades per Day")
    st.pyplot(fig)

with colB:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='classification', y='avg_trade_size', palette="magma", ax=ax2)
    ax2.set_title("Average Trade Size (USD)")
    st.pyplot(fig2)

st.divider()

# --- 4. BONUS: PREDICTIVE MODEL ---
st.subheader("🤖 Predictive Model: Profitability Predictor")
st.write("A Random Forest ML model predicting if a trader will be profitable today based on market sentiment and trade size.")

# Prep data for ML
df_ml = df.dropna().copy()
df_ml['is_profitable'] = (df_ml['daily_PnL'] > 0).astype(int) 

X = df_ml[['value', 'total_trades', 'avg_trade_size']] 
y = df_ml['is_profitable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"**Model Accuracy:** {accuracy * 100:.2f}%")
st.info("💡 **Insight:** The model leverages the 'Fear/Greed Value' alongside 'Trade Size' to estimate win probability.")
