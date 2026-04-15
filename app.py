import streamlit as st
import pandas as pd
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Primetrade Dashboard", layout="wide")

@st.cache_data
def load_data():
    # File names setup
    zip_name = "historical_data.zip"
    csv_name = "historical_data.csv"
    sent_name = "fear_greed_index.csv"
    
    # Path check (Root or Folder)
    folder_prefix = "Primetrade_Assignment/"
    
    # 1. LOAD SENTIMENT DATA
    if os.path.exists(sent_name):
        df_sent = pd.read_csv(sent_name)
    elif os.path.exists(folder_prefix + sent_name):
        df_sent = pd.read_csv(folder_prefix + sent_name)
    else:
        st.error(f"Missing {sent_name}")
        return None

    # 2. LOAD TRADES DATA (FROM ZIP)
    df_trades = None
    target_zip = zip_name if os.path.exists(zip_name) else (folder_prefix + zip_name if os.path.exists(folder_prefix + zip_name) else None)
    
    if target_zip:
        try:
            with zipfile.ZipFile(target_zip, 'r') as z:
                # Zip ke andar ki CSV file dhoond raha hai
                with z.open(csv_name) as f:
                    df_trades = pd.read_csv(f)
        except Exception as e:
            st.error(f"Error unzipping: {e}")
            return None
    else:
        st.error("historical_data.zip not found on GitHub!")
        return None

    # 3. PROCESSING
    if df_trades is not None and df_sent is not None:
        df_trades['date'] = pd.to_datetime(df_trades['Timestamp IST'], dayfirst=True).dt.strftime('%Y-%m-%d')
        df_sent['date'] = pd.to_datetime(df_sent['date']).dt.strftime('%Y-%m-%d')
        
        df_merged = pd.merge(df_trades, df_sent, on='date', how='inner')
        daily_metrics = df_merged.groupby(['date', 'Account']).agg(
            daily_PnL=('Closed PnL', 'sum'),
            total_trades=('Closed PnL', 'count'),
            avg_trade_size=('Size USD', 'mean')
        ).reset_index()
        
        return pd.merge(daily_metrics, df_sent[['date', 'classification', 'value']], on='date', how='left')
    
    return None

df = load_data()

if df is not None:
    st.title("📈 Primetrade Analysis Dashboard (Live)")
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades", len(df))
    c2.metric("Net PnL", f"${df['daily_PnL'].sum():,.2f}")
    c3.metric("Avg Sentiment", f"{df['value'].mean():.1f}")
    
    # Visualization
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### Trades per Sentiment")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='classification', y='total_trades', palette='viridis', ax=ax)
        st.pyplot(fig)
    with col_b:
        st.write("### Avg Trade Size (USD)")
        fig2, ax2 = plt.subplots()
        sns.barplot(data=df, x='classification', y='avg_trade_size', palette='magma', ax=ax2)
        st.pyplot(fig2)

    # ML Predictor
    st.divider()
    st.subheader("🤖 Profitability Predictor")
    df_ml = df.dropna()
    df_ml['target'] = (df_ml['daily_PnL'] > 0).astype(int)
    X = df_ml[['value', 'total_trades', 'avg_trade_size']]
    y = df_ml['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier().fit(X_train, y_train)
    st.success(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")
