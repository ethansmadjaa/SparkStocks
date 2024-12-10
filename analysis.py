# This is where we do our deep analysis of stock patterns and predictions
# We look at trends, seasonality, and try to understand what might happen next

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

from exploration import get_stock_data


def analyze_data(spark: SparkSession, ticker: str, days: int) -> None:
    """
    Perform and display comprehensive stock analysis including market sentiment.
    
    Args:
        spark: SparkSession instance
        ticker: Stock ticker symbol
        days: Number of days to analyze
    """
    st.header(f"Analysis for {ticker}")

    # Get the stock data
    df = get_stock_data(spark, ticker, days)

    # Create tabs for different types of analysis
    sentiment_tab, seasonal_tab, trend_tab = st.tabs([
        "Market Sentiment", "Seasonality", "Trend Analysis"
    ])

    with sentiment_tab:
        st.subheader("Overall Market Sentiment")

        # Calculate daily returns more efficiently
        window_spec = Window.orderBy("Date").partitionBy()
        df = df.withColumn(
            "daily_return",
            ((col("Close") - lag("Close", 1).over(window_spec)) / lag("Close", 1).over(window_spec)) * 100
        ).cache()

        # Convert to pandas for efficient calculations
        pdf = df.toPandas()

        # Calculate sentiment indicators using pandas
        recent_returns = pdf['daily_return'].head(10).dropna()
        avg_recent_return = recent_returns.mean() if not recent_returns.empty else 0

        # Volume analysis using pandas
        avg_volume = pdf['Volume'].mean()
        recent_volume = pdf['Volume'].head(5)
        avg_recent_volume = recent_volume.mean() if not recent_volume.empty else 0

        # Determine sentiment
        price_trend = "Bullish" if avg_recent_return > 0 else "Bearish"
        volume_trend = "Increasing" if avg_recent_volume > avg_volume else "Decreasing"

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Price Trend", price_trend)
        with col2:
            st.metric("Recent Return", f"{avg_recent_return:.2f}%")
        with col3:
            st.metric("Volume Trend", volume_trend)

        # Create sentiment chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Price Movement", "Trading Volume"),
            vertical_spacing=0.12
        )

        # Price chart
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

        # Volume chart
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
        seasonal_results = analyze_seasonality(df)
        seasonal_pdf = seasonal_results.toPandas()

        # Format the seasonal analysis results
        seasonal_pdf['month'] = pd.to_datetime(seasonal_pdf['month'], format='%m').dt.strftime('%B')
        st.write("Monthly Return Patterns:", seasonal_pdf)

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
            yaxis_title="Average Return (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with trend_tab:
        trend_results = analyze_trend_strength(df)
        trend_pdf = trend_results.toPandas()
        st.write("Trend Analysis Results:", trend_pdf)


def analyze_seasonality(df: DataFrame) -> DataFrame:
    # This helps us see if the stock has regular patterns
    # For example: Does it usually go up in December? Down in September?

    # Add month and day of week columns
    df = df.withColumn("month", month("Date"))
    df = df.withColumn("day_of_week", dayofweek("Date"))

    # Calculate average returns by month
    monthly_patterns = df.groupBy("month").agg(
        mean("daily_return").alias("avg_return"),
        stddev("daily_return").alias("return_volatility"),
        count("*").alias("days_counted")
    )

    return monthly_patterns


def analyze_trend_strength(df: DataFrame, window: int = 20) -> DataFrame:
    """
    Calculate trend strength using ADX (Average Directional Index).
    
    Args:
        df: DataFrame with OHLC data
        window: Period for calculations (default 20)
    
    Returns:
        DataFrame with trend indicators
    """
    # Create window specs with proper partitioning
    window_spec = Window.orderBy("Date").rowsBetween(-window, 0)
    lag_window = Window.orderBy("Date")

    # Calculate True Range
    df = df.withColumn(
        "high_low", col("High") - col("Low")
    ).withColumn(
        "high_close", abs(col("High") - lag("Close", 1).over(lag_window))
    ).withColumn(
        "low_close", abs(col("Low") - lag("Close", 1).over(lag_window))
    ).withColumn(
        "true_range",
        greatest(
            col("high_low"),
            col("high_close"),
            col("low_close")
        )
    )

    # Calculate Directional Movement
    df = df.withColumn(
        "plus_dm",
        when(
            (col("High") - lag("High", 1).over(lag_window)) >
            (lag("Low", 1).over(lag_window) - col("Low")),
            col("High") - lag("High", 1).over(lag_window)
        ).otherwise(lit(0.0))
    ).withColumn(
        "minus_dm",
        when(
            (lag("Low", 1).over(lag_window) - col("Low")) >
            (col("High") - lag("High", 1).over(lag_window)),
            lag("Low", 1).over(lag_window) - col("Low")
        ).otherwise(lit(0.0))
    )

    # Calculate smoothed averages
    df = df.withColumn(
        "smoothed_tr", avg("true_range").over(window_spec)
    ).withColumn(
        "smoothed_plus_dm", avg("plus_dm").over(window_spec)
    ).withColumn(
        "smoothed_minus_dm", avg("minus_dm").over(window_spec)
    )

    # Calculate directional indicators
    df = df.withColumn(
        "plus_di", (col("smoothed_plus_dm") / col("smoothed_tr")) * 100
    ).withColumn(
        "minus_di", (col("smoothed_minus_dm") / col("smoothed_tr")) * 100
    )

    # Calculate ADX
    df = df.withColumn(
        "dx",
        abs(col("plus_di") - col("minus_di")) /
        (col("plus_di") + col("minus_di")) * 100
    ).withColumn(
        "adx",
        avg("dx").over(window_spec)
    )

    # Select relevant columns and add trend interpretation
    result = df.select(
        "Date",
        "adx",
        "plus_di",
        "minus_di"
    ).withColumn(
        "trend_strength",
        when(col("adx") > 25, "Strong")
        .when(col("adx") > 15, "Moderate")
        .otherwise("Weak")
    ).withColumn(
        "trend_direction",
        when(col("plus_di") > col("minus_di"), "Bullish")
        .otherwise("Bearish")
    )

    return result.orderBy(desc("Date"))
