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

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Primetrade: Trader Performance & Sentiment Analysis")
st.markdown("An interactive dashboard analyzing how **Fear & Greed** sentiment impacts trader behavior and profitability on Hyperliquid.")

# --- 2. DATA LOADING ENGINE ---
@st.cache_data
def load_data():
    try:
        # Using Relative Paths for Cloud Deployment
        path_trades = "historical_data.csv"
        path_sent = "fear_greed_index.csv"
        
        df_trades = pd.read_csv(path_trades)
        df_sentiment = pd.read_csv(path_sent)
        
        # Robust Date Parsing
        df_trades['date'] = pd.to_datetime(df_trades['Timestamp IST'], dayfirst=True).dt.strftime('%Y-%m-%d')
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.strftime('%Y-%m-%d')
        
        # Merging Datasets
        df_merged = pd.merge(df_trades, df_sentiment, on='date', how='inner')
        
        # Calculating Daily Metrics per Account
        daily_metrics = df_merged.groupby(['date', 'Account']).agg(
            daily_PnL=('Closed PnL', 'sum'),
            total_trades=('Closed PnL', 'count'),
            avg_trade_size=('Size USD', 'mean')
        ).reset_index()
        
        # Adding Sentiment Classification back
        daily_metrics = pd.merge(daily_metrics, df_sentiment[['date', 'classification', 'value']], on='date', how='left')
        return daily_metrics
    except FileNotFoundError:
        st.error("❌ Data files not found. Please ensure 'historical_data.csv' and 'fear_greed_index.csv' are in the same directory.")
        return None

df = load_data()

if df is not None:
    # --- 3. EXECUTIVE KPI SUMMARY ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_pnl = df['daily_PnL'].sum()
    profitable_days_pct = (df['daily_PnL'] > 0).mean() * 100

    col1.metric("Total Trade-Account Days", len(df))
    col2.metric("Total Net PnL", f"${total_pnl:,.2f}")
    col3.metric("Avg Win Rate (Daily)", f"{profitable_days_pct:.1f}%")
    col4.metric("Avg Sentiment Score", f"{df['value'].mean():.1f}")

    st.divider()

    # --- 4. BEHAVIORAL ANALYSIS VISUALS ---
    st.subheader("📊 Behavioral Shifts: Fear vs Greed")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### Average Trades Frequency")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x='classification', y='total_trades', palette="viridis", ax=ax, ci=None)
        ax.set_ylabel("Avg Trades / Day")
        st.pyplot(fig)

    with chart_col2:
        st.markdown("##### Average Trade Size (Risk Appetite)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x='classification', y='avg_trade_size', palette="magma", ax=ax2, ci=None)
        ax2.set_ylabel("Avg Size (USD)")
        st.pyplot(fig2)

    st.divider()

    # --- 5. PREDICTIVE AI MODEL ---
    st.subheader("🤖 Bonus: Profitability Predictor (Random Forest)")
    st.write("This ML model predicts if a trader will be profitable based on market sentiment and trade frequency.")

    # Data Preparation for ML
    df_ml = df.dropna().copy()
    df_ml['is_profitable'] = (df_ml['daily_PnL'] > 0).astype(int) 

    features = ['value', 'total_trades', 'avg_trade_size']
    X = df_ml[features] 
    y = df_ml['is_profitable']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.success(f"**Model Accuracy:** {accuracy * 100:.2f}%")
    with m_col2:
        st.info("💡 **Strategy Insight:** Market sentiment combined with trade sizing is a significant predictor of success. Institutional 'Greed' periods often correlate with higher sizing but also higher volatility.")

st.markdown("---")
st.caption("Developed for Primetrade Data Science Internship Assignment | Powered by Streamlit & Scikit-Learn")
