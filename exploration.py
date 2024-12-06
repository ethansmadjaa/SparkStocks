from pyspark.sql import DataFrame
import streamlit as st
import yfinance as yf
from pyspark.sql.functions import col, count, min, max, mean, stddev, date_format
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance."""
    try:
        # Get data for the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def convert_to_spark_df(spark, pandas_df):
    """Convert pandas DataFrame to Spark DataFrame."""
    if pandas_df is not None:
        # Reset index to make date a column
        pandas_df = pandas_df.reset_index()
        return spark.createDataFrame(pandas_df)
    return None

def explore_data(spark, ticker: str):
    """Main function for data exploration."""
    st.subheader(f"📊 Exploring Data for {ticker}")
    
    # 1. Data Loading
    with st.spinner("Fetching data..."):
        pandas_df = fetch_stock_data(ticker)
        if pandas_df is None:
            return
        
        df = convert_to_spark_df(spark, pandas_df)
        
    # 2. Initial Data Overview
    st.write("### 📋 Data Overview")
    
    # Show basic information about the dataset
    row_count = df.count()
    col_count = len(df.columns)
    st.write(f"Dataset contains {row_count:,} rows and {col_count} columns.")
    
    # Display schema
    st.write("### 📑 Data Schema")
    st.code(df.schema.simpleString())
    
    # 3. Sample Data Display
    st.write("### 🔍 Sample Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("First few rows:")
        st.dataframe(df.limit(5).toPandas())
    
    with col2:
        st.write("Last few rows:")
        st.dataframe(df.orderBy(col("Date").desc()).limit(5).toPandas())
    
    # 4. Descriptive Statistics
    st.write("### 📈 Descriptive Statistics")
    
    # Calculate statistics for numeric columns
    numeric_stats = df.select([
        mean(c).alias(f"{c}_mean"),
        stddev(c).alias(f"{c}_stddev"),
        min(c).alias(f"{c}_min"),
        max(c).alias(f"{c}_max")
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']]).toPandas()
    
    # Display statistics in a more readable format
    stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
        'Open': [numeric_stats['Open_mean'][0], numeric_stats['Open_stddev'][0],
                numeric_stats['Open_min'][0], numeric_stats['Open_max'][0]],
        'Close': [numeric_stats['Close_mean'][0], numeric_stats['Close_stddev'][0],
                 numeric_stats['Close_min'][0], numeric_stats['Close_max'][0]],
        'Volume': [numeric_stats['Volume_mean'][0], numeric_stats['Volume_stddev'][0],
                  numeric_stats['Volume_min'][0], numeric_stats['Volume_max'][0]]
    })
    st.dataframe(stats_df)
    
    # 5. Time Series Visualization
    st.write("### 📉 Price History")
    
    # Convert Spark DataFrame back to pandas for plotting
    plot_df = df.toPandas()
    
    # Create interactive price chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Close'],
                            mode='lines',
                            name='Close Price'))
    fig.update_layout(title=f'{ticker} Stock Price Over Time',
                     xaxis_title='Date',
                     yaxis_title='Price (USD)',
                     hovermode='x unified')
    st.plotly_chart(fig)
    
    # 6. Volume Analysis
    st.write("### 📊 Trading Volume")
    
    # Create volume chart
    fig_volume = px.bar(plot_df, x='Date', y='Volume',
                       title=f'{ticker} Trading Volume Over Time')
    st.plotly_chart(fig_volume)
    
    # 7. Price Distribution
    st.write("### 🔄 Price Distribution")
    
    fig_dist = px.histogram(plot_df, x='Close',
                           title=f'Distribution of {ticker} Closing Prices',
                           nbins=50)
    st.plotly_chart(fig_dist)
    
    # 8. Data Quality Check
    st.write("### ✅ Data Quality Check")
    
    # Check for missing values
    missing_values = df.select([count(col(c)).alias(f"{c}_count") 
                              for c in df.columns]).toPandas()
    st.write("Missing values check:", "No missing values" if missing_values.all().all() 
             else "Contains missing values")
