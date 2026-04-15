import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Page Config
st.set_page_config(page_title="Primetrade AI Dashboard", layout="wide")

# --- 2. DATA LOADING (SIMPLE PATHS) ---
@st.cache_data
def load_data():
    try:
        # Files ab seedha main folder mein hain
        path_trades = "historical_data.csv"
        path_sent = "fear_greed_index.csv"
        
        df_trades = pd.read_csv(path_trades)
        df_sentiment = pd.read_csv(path_sent)
        
        # Date Conversion
        df_trades['date'] = pd.to_datetime(df_trades['Timestamp IST'], dayfirst=True).dt.strftime('%Y-%m-%d')
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.strftime('%Y-%m-%d')
        
        df_merged = pd.merge(df_trades, df_sentiment, on='date', how='inner')
        
        daily_metrics = df_merged.groupby(['date', 'Account']).agg(
            daily_PnL=('Closed PnL', 'sum'),
            total_trades=('Closed PnL', 'count'),
            avg_trade_size=('Size USD', 'mean')
        ).reset_index()
        
        daily_metrics = pd.merge(daily_metrics, df_sentiment[['date', 'classification', 'value']], on='date', how='left')
        return daily_metrics
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    st.title("📈 Primetrade Analysis Dashboard")
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades", len(df))
    c2.metric("Net PnL", f"${df['daily_PnL'].sum():,.2f}")
    c3.metric("Avg Sentiment", f"{df['value'].mean():.1f}")
    
    # Graphs
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='classification', y='total_trades', palette='viridis', ax=ax)
        st.pyplot(fig)
    with col_b:
        fig2, ax2 = plt.subplots()
        sns.barplot(data=df, x='classification', y='avg_trade_size', palette='magma', ax=ax2)
        st.pyplot(fig2)

    # ML Model
    st.divider()
    st.subheader("🤖 Profitability Predictor")
    df_ml = df.dropna()
    df_ml['target'] = (df_ml['daily_PnL'] > 0).astype(int)
    X = df_ml[['value', 'total_trades', 'avg_trade_size']]
    y = df_ml['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier().fit(X_train, y_train)
    st.success(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")
