from typing import Any

from pyspark.sql import DataFrame
import streamlit as st
from pyspark.sql.functions import (
    col, count, min, max, mean, stddev,
    lag, when, lit, percentile_approx, skewness,
    kurtosis, desc, sum, avg
)
from pyspark.sql.window import Window
from pyspark.sql.types import *
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta


def fetch_stock_data(spark, ticker: str, days: int | str = 365) -> Any | None:
    """Fetch stock data and return as Spark DataFrame."""
    try:
        end_date = datetime.now()

        if days is None:
            stock_data = yf.download(ticker, period="max")
        elif days == "ytd":
            start_date = datetime(end_date.year, 1, 1)
            stock_data = yf.download(ticker, start=start_date, end=end_date)
        else:
            start_date = end_date - timedelta(days=days)
            stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            st.error(f"No data available for {ticker}")
            return None

        # Define schema for Spark DataFrame
        schema = StructType([
            StructField("Date", DateType(), False),
            StructField("Open", DoubleType(), True),
            StructField("High", DoubleType(), True),
            StructField("Low", DoubleType(), True),
            StructField("Close", DoubleType(), True),
            StructField("Adj Close", DoubleType(), True),
            StructField("Volume", LongType(), True)
        ])

        # Convert to Spark DataFrame
        stock_data = stock_data.reset_index()
        spark_df = spark.createDataFrame(stock_data, schema=schema)

        st.success(f"Successfully fetched {spark_df.count()} days of data")
        return spark_df

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def prepare_data(df: DataFrame) -> DataFrame:
    """Prepare data with all necessary calculations using Spark."""
    # Window for sequential calculations with partitioning
    w = Window.partitionBy(lit(1)).orderBy("Date")  # Partition by constant for time series

    # Calculate returns and changes
    df = df.withColumn("prev_close", lag("Close").over(w))
    df = df.withColumn("daily_return",
                       when(col("prev_close").isNotNull(),
                            ((col("Close") - col("prev_close")) / col("prev_close") * 100)
                            ).otherwise(None)
                       )

    # Calculate volume changes
    df = df.withColumn("prev_volume", lag("Volume").over(w))
    df = df.withColumn("volume_change",
                       when(col("prev_volume").isNotNull(),
                            col("Volume") - col("prev_volume")
                            ).otherwise(None)
                       )

    # Calculate moving averages with optimized windows
    w_ma = Window.partitionBy(lit(1)).orderBy("Date").rowsBetween(-199, 0)  # Maximum window size needed

    df = df.withColumn("MA20", avg("Close").over(w_ma.rowsBetween(-19, 0)))
    df = df.withColumn("MA50", avg("Close").over(w_ma.rowsBetween(-49, 0)))
    df = df.withColumn("MA200", avg("Close").over(w_ma))

    # Calculate volatility windows
    df = df.withColumn("volatility_20d",
                       stddev("daily_return").over(w_ma.rowsBetween(-19, 0))
                       )

    # Repartition for better performance
    df = df.repartition("Date")

    return df


def calculate_summary_stats(df: DataFrame) -> dict:
    """Calculate summary statistics using Spark."""
    stats = df.select([
        mean("Close").alias("mean_price"),
        stddev("Close").alias("std_price"),
        min("Close").alias("min_price"),
        max("Close").alias("max_price"),
        mean("daily_return").alias("avg_return"),
        stddev("daily_return").alias("volatility"),
        skewness("Close").alias("skewness"),
        kurtosis("Close").alias("kurtosis"),
        percentile_approx("Close", 0.25).alias("q1"),
        percentile_approx("Close", 0.5).alias("median"),
        percentile_approx("Close", 0.75).alias("q3"),
        avg("Volume").alias("avg_volume"),
        sum(when(col("volume_change") > 0, 1)).alias("increasing_volume_days"),
        count("*").alias("total_days")
    ]).collect()[0]

    return stats


def display_price_analysis(df: DataFrame, ticker: str):
    """Display price analysis using Spark calculations."""
    # Get latest price info
    latest = df.orderBy(desc("Date")).first()

    # Calculate price changes
    daily_change = ((latest.Close - latest.prev_close) / latest.prev_close * 100)

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${latest.Close:.2f}",
                  f"{daily_change:+.2f}%")
    with col2:
        st.metric("Daily Volume", f"{latest.Volume:,}",
                  f"{latest.volume_change:+,}")
    with col3:
        st.metric("20-Day Volatility",
                  f"{latest.volatility_20d:.2f}%")


def display_technical_indicators(df: DataFrame):
    """Display technical indicators calculated using Spark."""
    latest = df.orderBy(desc("Date")).first()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MA20", f"${latest.MA20:.2f}",
                  f"{((latest.Close - latest.MA20) / latest.MA20 * 100):+.2f}%")
    with col2:
        st.metric("MA50", f"${latest.MA50:.2f}",
                  f"{((latest.Close - latest.MA50) / latest.MA50 * 100):+.2f}%")
    with col3:
        st.metric("MA200", f"${latest.MA200:.2f}",
                  f"{((latest.Close - latest.MA200) / latest.MA200 * 100):+.2f}%")


def create_price_chart(df: DataFrame, ticker: str) -> go.Figure:
    """Create interactive price chart with moving averages and volume."""
    # Convert necessary data to pandas for plotting
    plot_data = df.select(
        'Date', 'Close', 'Volume', 'MA20', 'MA50', 'MA200', 'volume_change'
    ).orderBy('Date').toPandas()

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(
        go.Scatter(
            x=plot_data['Date'],
            y=plot_data['Close'],
            name='Price',
            line=dict(color='#2962FF', width=2),
            hovertemplate="<br>".join([
                "Date: %{x}",
                "Price: $%{y:.2f}",
                "<extra></extra>"
            ])
        )
    )

    # Add moving averages
    ma_colors = {'MA20': '#FF6D00', 'MA50': '#2E7D32', 'MA200': '#D32F2F'}

    for ma in ['MA20', 'MA50', 'MA200']:
        fig.add_trace(
            go.Scatter(
                x=plot_data['Date'],
                y=plot_data[ma],
                name=ma,
                line=dict(color=ma_colors[ma], width=1, dash='dot'),
                hovertemplate=f"{ma}: $%{{y:.2f}}<extra></extra>"
            )
        )

    # Add volume bars to secondary y-axis
    colors = ['rgba(0, 255, 0, 0.3)' if x >= 0 else 'rgba(255, 0, 0, 0.3)'
              for x in plot_data['volume_change']]

    fig.add_trace(
        go.Bar(
            x=plot_data['Date'],
            y=plot_data['Volume'],
            name='Volume',
            marker_color=colors,
            yaxis='y2',
            hovertemplate="Volume: %{y:,.0f}<extra></extra>"
        )
    )

    # Update layout with secondary y-axis
    fig.update_layout(
        title=dict(
            text=f'{ticker} Price History with Moving Averages',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Date',
        yaxis_title='Price ($)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            side='left'
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig


def create_volume_chart(df: DataFrame, ticker: str) -> go.Figure:
    """Create interactive volume chart with color-coded bars."""
    # Get necessary data
    plot_data = df.select(
        'Date', 'Volume', 'volume_change'
    ).toPandas()

    colors = ['rgba(0, 255, 0, 0.5)' if x >= 0 else 'rgba(255, 0, 0, 0.5)'
              for x in plot_data['volume_change']]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_data['Date'],
            y=plot_data['Volume'],
            marker_color=colors,
            name='Volume'
        )
    )

    fig.update_layout(
        title=f'{ticker} Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        hovermode='x unified'
    )

    return fig


def create_returns_distribution(df: DataFrame, ticker: str) -> go.Figure:
    """Create distribution plot of daily returns."""
    plot_data = df.select('daily_return').toPandas()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=plot_data['daily_return'],
            nbinsx=50,
            name='Daily Returns',
            marker_color='rgba(0, 123, 255, 0.6)'
        )
    )

    fig.update_layout(
        title=f'{ticker} Daily Returns Distribution',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        showlegend=False
    )

    return fig


def explore_data(spark, ticker: str, days: int = 365):
    """Main function for data exploration using Spark."""
    st.subheader(f"📊 Exploring Data for {ticker}")

    # 1. Data Loading and Processing
    with st.spinner("Fetching and processing data..."):
        df = fetch_stock_data(spark, ticker, days)
        if df is None:
            return

        # Prepare data with all calculations
        df = prepare_data(df)

        # Cache the DataFrame as we'll be using it multiple times
        df.cache()

    # 2. Current Price Analysis
    st.write("### 📈 Price Analysis")
    display_price_analysis(df, ticker)

    # 3. Technical Indicators
    st.write("### 📊 Technical Indicators")
    display_technical_indicators(df)

    # 4. Statistical Summary
    st.write("### 📉 Statistical Summary")
    stats = calculate_summary_stats(df)

    # Display statistics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Price", f"${stats.mean_price:.2f}")
        st.metric("Volatility", f"{stats.volatility:.2f}%")
    with col2:
        st.metric("Minimum", f"${stats.min_price:.2f}")
        st.metric("Maximum", f"${stats.max_price:.2f}")
    with col3:
        st.metric("Skewness", f"{stats.skewness:.2f}")
        st.metric("Kurtosis", f"{stats.kurtosis:.2f}")
    with col4:
        st.metric("Avg Volume", f"{stats.avg_volume:,.0f}")
        st.metric("Up Volume Days",
                  f"{(stats.increasing_volume_days / stats.total_days * 100):.1f}%")

    # 5. Data Sample
    st.write("### 🔍 Data Sample")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Recent Data:")
        st.dataframe(df.orderBy(desc("Date")).limit(5).toPandas())
    with col2:
        st.write("Summary Statistics:")
        st.dataframe(df.select("Close", "Volume", "daily_return")
                     .summary().toPandas())

    # Add Charts Section
    st.write("### 📊 Interactive Charts")

    # Price Chart with Moving Averages
    st.plotly_chart(
        create_price_chart(df, ticker),
        use_container_width=True
    )

    # Volume Analysis
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            create_volume_chart(df, ticker),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            create_returns_distribution(df, ticker),
            use_container_width=True
        )

    # Uncache the DataFrame
    df.unpersist()
