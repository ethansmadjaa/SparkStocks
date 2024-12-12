# This is where we do our deep analysis of stock patterns and predictions
# We look at trends, seasonality, and try to understand what might happen next

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

from utils.data_utils import get_stock_data
from utils.preprocessing_utils import (
    calculate_rsi,
    calculate_macd,
    calculate_trading_signals
)

def analyze_data(spark: SparkSession, ticker: str, days: int = 365):
    """
    Perform deep analysis of stock patterns and predictions.
    """
    st.write("## Advanced Stock Analysis")

    # Analysis Methods Explanation
    with st.expander("Understanding Analysis Methods"):
        st.write("""
        ### Analysis Techniques Guide
        
        #### Price Analysis
        - **Support Levels**: Price levels where buying pressure typically increases
        - **Resistance Levels**: Price levels where selling pressure typically increases
        - **Price Patterns**: Recurring price movements that may indicate future direction
        
        #### Volume Analysis
        Understanding trading activity:
        - **Volume Trends**: Shows market interest and participation
        - **Volume Confirmation**: Higher volume validates price movements
        - **Volume Divergence**: When volume doesn't confirm price movement
        """)

    # Get the stock data with all indicators
    df = get_stock_data(spark, ticker, days)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_trading_signals(df)

    # Create tabs for different types of analysis
    sentiment_tab, seasonal_tab = st.tabs([
        "Market Sentiment", "Seasonality Analysis"
    ])

    with sentiment_tab:
        st.subheader("Overall Market Sentiment")

        # Calculate daily returns
        window_spec = Window.orderBy("Date").partitionBy()
        df = df.withColumn(
            "daily_return",
            ((col("Close") - lag("Close", 1).over(window_spec)) / lag("Close", 1).over(window_spec)) * 100
        ).cache()

        # Convert to pandas for calculations
        pdf = df.toPandas()

        # Calculate sentiment metrics
        recent_returns = pdf['daily_return'].head(10).dropna()
        avg_recent_return = recent_returns.mean() if not recent_returns.empty else 0
        avg_volume = pdf['Volume'].mean()
        recent_volume = pdf['Volume'].head(5)
        avg_recent_volume = recent_volume.mean() if not recent_volume.empty else 0

        # Determine market sentiment
        price_trend = "Bullish" if avg_recent_return > 0 else "Bearish"
        volume_trend = "Increasing" if avg_recent_volume > avg_volume else "Decreasing"

        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Price Trend", price_trend)
        with col2:
            st.metric("Recent Return", f"{avg_recent_return:.2f}%")
        with col3:
            st.metric("Volume Trend", volume_trend)

        # Create price and volume chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Price Movement", "Trading Volume"),
            vertical_spacing=0.12
        )

        # Add price candlesticks
        fig.add_trace(
            go.Candlestick(
                x=pdf['Date'],
                open=pdf['Open'],
                high=pdf['High'],
                low=pdf['Low'],
                close=pdf['Close'],
                name="Price"
            ),
            row=1, col=1
        )

        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=pdf['Date'],
                y=pdf['Volume'],
                name="Volume"
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Price and Volume Analysis",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume")
        )

        st.plotly_chart(fig, use_container_width=True)

    with seasonal_tab:
        st.subheader("Seasonal Analysis")
        seasonal_results = analyze_seasonality(df)
        seasonal_pdf = seasonal_results.toPandas()

        # Format seasonal analysis
        seasonal_pdf['month'] = pd.to_datetime(seasonal_pdf['month'], format='%m').dt.strftime('%B')
        
        # Display seasonal patterns
        st.write("### Monthly Performance Patterns")
        st.write("""
        This analysis shows how the stock typically performs in different months.
        - Higher returns indicate historically strong months
        - Higher volatility indicates more unpredictable months
        """)
        
        st.dataframe(seasonal_pdf)

        # Plot seasonal patterns
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=seasonal_pdf['month'],
            y=seasonal_pdf['avg_return'],
            name="Average Return"
        ))
        fig.update_layout(
            title="Monthly Return Patterns",
            xaxis_title="Month",
            yaxis_title="Average Return (%)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    return df

def analyze_seasonality(df: DataFrame) -> DataFrame:
    """Calculate seasonal patterns in stock performance."""
    # Add month column
    df = df.withColumn("month", month("Date"))

    # Calculate monthly patterns
    monthly_patterns = df.groupBy("month").agg(
        mean("daily_return").alias("avg_return"),
        stddev("daily_return").alias("return_volatility"),
        count("*").alias("days_counted")
    )

    return monthly_patterns
