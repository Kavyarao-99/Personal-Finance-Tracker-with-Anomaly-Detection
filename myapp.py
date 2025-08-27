import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy import stats
import numpy as np

# Page config
st.set_page_config(page_title="Finance Tracker", layout="wide")

# Title
st.title("💰 Personal Finance Tracker with Anomaly Detection")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("personal_finance_clean.csv", parse_dates=['Date'])

df = load_data()

# Add Z-score & Isolation Forest anomalies if not present
if 'Amount_Zscore' not in df.columns:
    df['Amount_Zscore'] = np.abs(stats.zscore(df['Amount']))
if 'Anomaly_Zscore' not in df.columns:
    df['Anomaly_Zscore'] = df['Amount_Zscore'] > 3
if 'Anomaly_ISO' not in df.columns:
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_ISO'] = model.fit_predict(df[['Amount']])
    df['Anomaly_ISO'] = df['Anomaly_ISO'].map({1: 0, -1: 1})

# Sidebar filters
st.sidebar.header("Filters")
selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=[df['Date'].min(), df['Date'].max()],
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

# Apply filters
filtered_df = df[
    (df['Category'].isin(selected_categories)) &
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1]))
]

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Overview", "Anomaly Detection", "Category Analysis"])

with tab1:
    st.header("Financial Overview")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"${filtered_df[filtered_df['Amount'] > 0]['Amount'].sum():,.2f}")
    col2.metric("Total Expenses", f"${abs(filtered_df[filtered_df['Amount'] < 0]['Amount'].sum()):,.2f}")
    col3.metric("Net Balance", f"${filtered_df['Amount'].sum():,.2f}")
    
    # Interactive time series chart
    st.subheader("Transactions Over Time")
    fig = px.scatter(
        filtered_df,
        x='Date',
        y='Amount',
        color='Category',
        hover_data=['Description', 'Amount_Zscore'],
        title="Transactions Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Anomaly Detection")
    
    
    # Select method
    method = st.radio("Detection Method", ["Z-Score", "Isolation Forest"], horizontal=True)
    
    if method == "Z-Score":
        filtered_df['Amount_Zscore'] = np.abs(stats.zscore(filtered_df['Amount']))
        filtered_df['Anomaly'] = filtered_df['Amount_Zscore'] > 3
        threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.5)
        filtered_df['Anomaly'] = filtered_df['Amount_Zscore'] > threshold
    else:
        contamination = st.slider("Contamination Rate", 0.01, 0.2, 0.05, 0.01)
        model = IsolationForest(contamination=contamination, random_state=42)
        filtered_df['Anomaly'] = model.fit_predict(filtered_df[['Amount']])
        filtered_df['Anomaly'] = filtered_df['Anomaly'].map({1: 0, -1: 1})

    #Show anomalies
    st.subheader("Detected Anomalies")
    st.dataframe(
        filtered_df[filtered_df['Anomaly'] == 1][['Date', 'Description', 'Amount', 'Category']],
        height=300
    )

    # Plot anomalies
    st.subheader("Anomaly Visualization")
    
    fig = px.scatter(
        filtered_df,
        x='Date',
        y='Amount',
        color='Anomaly',
        color_discrete_map={0: "blue", 1: "red"},
        hover_data=['Description', 'Category', 'Amount_Zscore'],
        title="Transactions with Anomalies Highlighted"
        )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- New: Category vs Method Comparison ---
    st.subheader("Category-wise Anomaly Count Comparison")
    if not filtered_df.empty:
        comparison_df = filtered_df.copy()

        # Calculate anomalies for BOTH methods regardless of selection
        comparison_df['Zscore_Anomaly'] = comparison_df['Amount_Zscore'] > 3
        iso_model = IsolationForest(contamination=0.05, random_state=42)
        comparison_df['ISO_Anomaly'] = iso_model.fit_predict(comparison_df[['Amount']])
        comparison_df['ISO_Anomaly'] = comparison_df['ISO_Anomaly'].map({1: 0, -1: 1})

        # Group and reshape
        category_counts = comparison_df.groupby('Category').agg(
            Zscore_Anomalies=('Zscore_Anomaly', 'sum'),
            ISO_Anomalies=('ISO_Anomaly', 'sum')
        ).reset_index()

        # Melt for plotting
        category_counts_melted = category_counts.melt(
            id_vars='Category',
            value_vars=['Zscore_Anomalies', 'ISO_Anomalies'],
            var_name='Method',
            value_name='Count'
        )

        # Plot
        fig_compare = px.bar(
            category_counts_melted,
            x='Category',
            y='Count',
            color='Method',
            barmode='group',
            title="Z-Score vs Isolation Forest - Anomalies per Category"
        )
        st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.warning("No data available for method comparison.")
        
with tab3:
    st.header("Category Analysis")
    
    # Spending by category
    st.subheader("Spending by Category")
    category_spending = filtered_df[filtered_df['Amount'] < 0] \
        .groupby('Category')['Amount'] \
        .sum().abs().reset_index()
    fig = px.bar(
        category_spending,
        x='Amount',
        y='Category',
        orientation='h',
        color='Amount',
        title="Total Spending by Category"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomalies by category
    if 'Anomaly' in filtered_df.columns:
        st.subheader("Anomalies by Category")
        anomaly_counts = filtered_df[filtered_df['Anomaly'] == 1] \
            .groupby('Category') \
            .size().reset_index(name='Count')
        fig = px.bar(
            anomaly_counts,
            x='Category',
            y='Count',
            color='Count',
            title="Anomalies by Category"
        )
        st.plotly_chart(fig, use_container_width=True)
       

# Download button
st.sidebar.download_button(
    label="Download Processed Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_finance_data.csv",
    mime="text/csv"
)
