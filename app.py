import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Page Configuration
st.set_page_config(page_title="Primetrade AI Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Primetrade: Trader Performance & Sentiment Analysis")
st.markdown("An interactive dashboard analyzing how **Fear & Greed** sentiment impacts trader behavior and profitability.")

# --- 2. DATA LOADING ENGINE ---
@st.cache_data
def load_data():
    try:
        # FOLDER PATH ADDED - Based on your GitHub structure
        path_trades = "Primetrade_Assignment/historical_data.csv"
        path_sent = "Primetrade_Assignment/fear_greed_index.csv"
        
        # Load CSVs
        df_trades = pd.read_csv(path_trades)
        df_sentiment = pd.read_csv(path_sent)
        
        # Date Conversion logic
        df_trades['date'] = pd.to_datetime(df_trades['Timestamp IST'], dayfirst=True).dt.strftime('%Y-%m-%d')
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.strftime('%Y-%m-%d')
        
        # Merge
        df_merged = pd.merge(df_trades, df_sentiment, on='date', how='inner')
        
        # Grouping
        daily_metrics = df_merged.groupby(['date', 'Account']).agg(
            daily_PnL=('Closed PnL', 'sum'),
            total_trades=('Closed PnL', 'count'),
            avg_trade_size=('Size USD', 'mean')
        ).reset_index()
        
        # Merge sentiment info back
        daily_metrics = pd.merge(daily_metrics, df_sentiment[['date', 'classification', 'value']], on='date', how='left')
        return daily_metrics
        
    except FileNotFoundError:
        st.error(f"❌ Could not find files at: {path_trades}")
        st.info("Check if your folder name is exactly 'Primetrade_Assignment' and files are inside it.")
        return None

df = load_data()

if df is not None:
    # --- 3. KPI SUMMARY ---
    col1, col2, col3, col4 = st.columns(4)
    total_pnl = df['daily_PnL'].sum()
    profitable_days_pct = (df['daily_PnL'] > 0).mean() * 100

    col1.metric("Total Trade Days", len(df))
    col2.metric("Total Net PnL", f"${total_pnl:,.2f}")
    col3.metric("Avg Win Rate", f"{profitable_days_pct:.1f}%")
    col4.metric("Avg Sentiment Score", f"{df['value'].mean():.1f}")

    st.divider()

    # --- 4. VISUALIZATIONS ---
    st.subheader("📊 Behavioral Shifts: Fear vs Greed")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Avg Trades Frequency")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x='classification', y='total_trades', palette="viridis", ax=ax, ci=None)
        st.pyplot(fig)

    with c2:
        st.markdown("##### Avg Trade Size (USD)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x='classification', y='avg_trade_size', palette="magma", ax=ax2, ci=None)
        st.pyplot(fig2)

    st.divider()

    # --- 5. AI MODEL ---
    st.subheader("🤖 Profitability Predictor")
    df_ml = df.dropna().copy()
    df_ml['is_profitable'] = (df_ml['daily_PnL'] > 0).astype(int) 
    
    X = df_ml[['value', 'total_trades', 'avg_trade_size']] 
    y = df_ml['is_profitable']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    st.success(f"**Model Accuracy:** {acc * 100:.2f}%")
    st.info("💡 Insight: Traders tend to take larger, riskier positions during 'Greed' phases.")

st.caption("Developed for Primetrade Assignment")
