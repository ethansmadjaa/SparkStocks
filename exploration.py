from datetime import datetime, timedelta

import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, date_trunc, desc, first, lag,
    mean, stddev, min, max, datediff, sum, when,
    corr, percentile_approx, greatest, array,
    abs as spark_abs, coalesce, lit
)
from pyspark.sql.types import *
from pyspark.sql.window import Window

from utils.constants import STOCK_CATEGORIES


def get_stock_data(spark: SparkSession, ticker: str, days: int = 365) -> DataFrame:
    """Fetch stock data and convert to Spark DataFrame with optimized configuration."""
    # Configure Spark for better performance
    spark.conf.set("spark.sql.shuffle.partitions", "8")  # Adjust based on your data size
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")

    # Get data from yfinance
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Define schema for consistency
    schema = StructType([
        StructField("Date", TimestampType(), False),
        StructField("Open", DoubleType(), True),
        StructField("High", DoubleType(), True),
        StructField("Low", DoubleType(), True),
        StructField("Close", DoubleType(), True),
        StructField("Volume", LongType(), True),
        StructField("Dividends", DoubleType(), True),
        StructField("Stock_Splits", DoubleType(), True)
    ])

    # Convert to Spark DataFrame
    pdf = stock.history(start=start_date, end=end_date)
    df = spark.createDataFrame(pdf.reset_index(), schema=schema)

    # Cache the DataFrame since we'll be using it multiple times
    return df.cache()


def analyze_data_frequency(df: DataFrame) -> str:
    # This helps us understand how often we get data points
    # For example: daily data, weekly data, etc.
    # We figure this out by looking at the gaps between dates

    # Look at dates within each month
    window_spec = Window.orderBy("Date") \
        .partitionBy(date_trunc("month", col("Date")))

    # Calculate how many days between each data point
    df_with_diff = df.withColumn("date_diff",
                                 datediff(col("Date"), lag("Date", 1).over(window_spec)))

    # Find the most common gap between data points
    mode_freq = df_with_diff.groupBy("date_diff") \
        .count() \
        .orderBy(desc("count")) \
        .first()

    # Convert the number of days to a human-readable frequency
    freq_mapping = {
        1: "Daily",  # Data every day
        7: "Weekly",  # Data every week
        30: "Monthly",  # Data every month
        365: "Yearly"  # Data every year
    }
    return freq_mapping.get(mode_freq["date_diff"], f"Custom ({mode_freq['date_diff']} days)")


def calculate_basic_stats(df: DataFrame) -> DataFrame:
    # Get basic statistics for all our numeric columns
    # Things like average price, highest price, lowest price, etc.

    # Find all the numeric columns in our data
    numeric_cols = [f.name for f in df.schema.fields
                    if isinstance(f.dataType, (DoubleType, LongType))]

    # For each numeric column, calculate:
    # - Mean (average)
    # - Standard deviation (how much the values spread out)
    # - Minimum value
    # - Maximum value
    stats_expressions = []
    for col_name in numeric_cols:
        stats_expressions.extend([
            mean(col_name).alias(f"{col_name}_mean"),
            stddev(col_name).alias(f"{col_name}_stddev"),
            min(col_name).alias(f"{col_name}_min"),
            max(col_name).alias(f"{col_name}_max")
        ])

    stats = df.select(stats_expressions)
    return stats


def calculate_returns(df: DataFrame) -> DataFrame:
    # Calculate how much money you would have made (or lost)
    # We look at different time periods: daily, weekly, and monthly

    # Split the data by date to process it efficiently
    df = df.repartition("Date")

    # Daily returns: how much the price changed within one day
    df = df.withColumn("daily_return",
                       (col("Close") - col("Open")) / col("Open") * 100)

    # Weekly returns: how much the price changed over 5 trading days
    window_week = Window.orderBy("Date") \
        .rowsBetween(-5, 0) \
        .partitionBy(date_trunc("week", col("Date")))

    df = df.withColumn("weekly_return",
                       ((col("Close") - first("Close").over(window_week)) /
                        first("Close").over(window_week) * 100))

    # Monthly returns: how much the price changed over about 21 trading days
    window_month = Window.orderBy("Date") \
        .rowsBetween(-21, 0) \
        .partitionBy(date_trunc("month", col("Date")))

    df = df.withColumn("monthly_return",
                       ((col("Close") - first("Close").over(window_month)) /
                        first("Close").over(window_month) * 100))

    return df


def plot_stock_price_history(df: DataFrame, ticker: str):
    # This creates a candlestick chart - it's the classic way to view stock prices
    # Each candle shows us 4 things:
    # - Open price (where the box starts)
    # - Close price (where the box ends)
    # - High price (the line above the box)
    # - Low price (the line below the box)

    # First, convert our Spark DataFrame to pandas for plotting
    pdf = df.toPandas()

    # Create a figure with two parts:
    # 1. Price chart on top (70% of height)
    # 2. Volume chart below (30% of height)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,  # Both charts share the same x-axis (dates)
        vertical_spacing=0.03,  # Small gap between charts
        subplot_titles=(f'{ticker} Stock Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )

    # Add the candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=pdf['Date'],
            open=pdf['Open'],
            high=pdf['High'],
            low=pdf['Low'],
            close=pdf['Close'],
            name='OHLC'  # Open-High-Low-Close
        ),
        row=1, col=1
    )

    # Add the volume bars below
    fig.add_trace(
        go.Bar(
            x=pdf['Date'],
            y=pdf['Volume'],
            name='Volume'
        ),
        row=2, col=1
    )

    # Make it look nice
    fig.update_layout(
        title=f'{ticker} Stock Price History',
        yaxis_title='Stock Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,  # Hide the range slider
        height=800  # Make it big enough to see details
    )

    return fig


def plot_returns_analysis(df: DataFrame):
    # This shows us how the stock's returns (profits/losses) look over time
    # We look at daily, weekly, and monthly returns to see different patterns

    # Get just the columns we need
    pdf = df.select('Date', 'daily_return', 'weekly_return', 'monthly_return').toPandas()

    # Create two plots:
    # 1. Returns over time (line chart)
    # 2. Returns distribution (box plot)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Returns Over Time', 'Returns Distribution (Box Plot)'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )

    # Add returns over time - three lines for different timeframes
    # Thicker lines for longer timeframes (they're usually smoother)
    fig.add_trace(
        go.Scatter(
            x=pdf['Date'],
            y=pdf['daily_return'],
            name='Daily Returns',
            mode='lines',
            line=dict(color='blue', width=1)  # Thin line for daily
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=pdf['Date'],
            y=pdf['weekly_return'],
            name='Weekly Returns',
            mode='lines',
            line=dict(color='green', width=1.5)  # Medium line for weekly
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=pdf['Date'],
            y=pdf['monthly_return'],
            name='Monthly Returns',
            mode='lines',
            line=dict(color='red', width=2)  # Thick line for monthly
        ),
        row=1, col=1
    )

    # Add box plots to show the distribution of returns
    # This helps us see:
    # - Typical return ranges (the boxes)
    # - Unusual returns (the dots outside the whiskers)
    # - If returns are symmetric (box centered around 0)
    colors = ['blue', 'green', 'red']
    for return_type, color in zip(['daily_return', 'weekly_return', 'monthly_return'], colors):
        fig.add_trace(
            go.Box(
                y=pdf[return_type],
                name=return_type.split('_')[0].capitalize(),
                boxpoints='outliers',  # Show points outside whiskers
                jitter=0.3,  # Spread out the outlier points
                whiskerwidth=0.2,
                marker_color=color,
                line_width=1
            ),
            row=2, col=1
        )

    # Make it look nice
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Label the axes
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)

    return fig


def calculate_moving_average(df: DataFrame, column_name: str, window_size: int) -> DataFrame:
    """
    Calculate moving average for a specified column.
    
    Args:
        df: Spark DataFrame
        column_name: Name of the column to calculate MA for
        window_size: Number of periods to consider for moving average
    
    Returns:
        DataFrame with new MA column added
    """
    # Create window specification for moving average
    window_spec = Window.orderBy("Date") \
        .rowsBetween(-(window_size - 1), 0)

    # Calculate moving average
    ma_column_name = f"{column_name}_MA_{window_size}"
    df = df.withColumn(
        ma_column_name,
        mean(col(column_name)).over(window_spec)
    )

    return df


def plot_moving_averages(df: DataFrame, ticker: str, column_name: str, ma_periods: list):
    """Create interactive plot with moving averages."""
    # Convert necessary columns to pandas
    plot_cols = ["Date", column_name] + [f"{column_name}_MA_{period}" for period in ma_periods]
    pdf = df.select(plot_cols).toPandas()

    # Create figure
    fig = go.Figure()

    # Add original price line
    fig.add_trace(
        go.Scatter(
            x=pdf['Date'],
            y=pdf[column_name],
            name=column_name,
            line=dict(color='black', width=1)
        )
    )

    # Add moving averages with different colors
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for period, color in zip(ma_periods, colors):
        ma_col = f"{column_name}_MA_{period}"
        fig.add_trace(
            go.Scatter(
                x=pdf['Date'],
                y=pdf[ma_col],
                name=f'{period}-day MA',
                line=dict(color=color, width=1.5)
            )
        )

    # Update layout
    fig.update_layout(
        title=f'{ticker} - {column_name} with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def analyze_ma_signals(df: DataFrame, short_period: int = 5, long_period: int = 20) -> DataFrame:
    # This function looks for trading signals based on moving average crossovers
    # We use two moving averages:
    # - Short period (default 5 days) for quick reactions to price changes
    # - Long period (default 20 days) for identifying the overall trend

    # First, calculate both moving averages if they don't exist
    if f"Close_MA_{short_period}" not in df.columns:
        df = calculate_moving_average(df, "Close", short_period)
    if f"Close_MA_{long_period}" not in df.columns:
        df = calculate_moving_average(df, "Close", long_period)

    # Look for crossovers
    # Golden Cross: Short MA crosses above Long MA (bullish signal)
    # Death Cross: Short MA crosses below Long MA (bearish signal)
    df = df.withColumn(
        "signal",
        when(
            (col(f"Close_MA_{short_period}") > col(f"Close_MA_{long_period}")) &
            (lag(f"Close_MA_{short_period}", 1).over(Window.orderBy("Date")) <=
             lag(f"Close_MA_{long_period}", 1).over(Window.orderBy("Date"))),
            "Golden Cross (Buy)"
        ).when(
            (col(f"Close_MA_{short_period}") < col(f"Close_MA_{long_period}")) &
            (lag(f"Close_MA_{short_period}", 1).over(Window.orderBy("Date")) >=
             lag(f"Close_MA_{long_period}", 1).over(Window.orderBy("Date"))),
            "Death Cross (Sell)"
        ).otherwise("No Signal")
    )

    return df


def calculate_stock_correlation(spark: SparkSession, ticker1: str, ticker2: str, days: int) -> dict:
    # This helps us see how two stocks move together
    # High correlation means they tend to move in the same direction
    # Low or negative correlation means they move independently or oppositely

    # Get data for both stocks
    df1 = get_stock_data(spark, ticker1, days)
    df2 = get_stock_data(spark, ticker2, days)

    # Join the dataframes on date to align the prices
    df_joined = df1.join(
        df2,
        on="Date",
        how="inner"  # Only keep dates where we have data for both stocks
    ).select(
        df1["Date"],
        df1["Close"].alias("close1"),
        df2["Close"].alias("close2"),
        df1["Volume"].alias("volume1"),
        df2["Volume"].alias("volume2"),
        df1["High"].alias("high1"),
        df2["High"].alias("high2"),
        df1["Low"].alias("low1"),
        df2["Low"].alias("low2")
    )

    # Calculate correlations for different metrics
    correlations = df_joined.select(
        corr("close1", "close2").alias("Close"),
        corr("volume1", "volume2").alias("Volume"),
        corr("high1", "high2").alias("High"),
        corr("low1", "low2").alias("Low")
    ).first()

    return {
        "Close": correlations["Close"] or 0,  # Handle None values
        "Volume": correlations["Volume"] or 0,
        "High": correlations["High"] or 0,
        "Low": correlations["Low"] or 0
    }


def plot_correlation_comparison(df1: DataFrame, df2: DataFrame, ticker1: str, ticker2: str):
    # This shows how two stocks move relative to each other
    # We normalize the prices to start at 100 to make comparison easier

    # Convert both dataframes to pandas
    pdf1 = df1.select("Date", "Close").toPandas()
    pdf2 = df2.select("Date", "Close").toPandas()

    # Normalize prices to start at 100
    pdf1["Normalized"] = pdf1["Close"] / pdf1["Close"].iloc[0] * 100
    pdf2["Normalized"] = pdf2["Close"] / pdf2["Close"].iloc[0] * 100

    # Create the comparison plot
    fig = go.Figure()

    # Add line for first stock
    fig.add_trace(
        go.Scatter(
            x=pdf1["Date"],
            y=pdf1["Normalized"],
            name=ticker1,
            line=dict(color='blue', width=2)
        )
    )

    # Add line for second stock
    fig.add_trace(
        go.Scatter(
            x=pdf2["Date"],
            y=pdf2["Normalized"],
            name=ticker2,
            line=dict(color='red', width=2)
        )
    )

    # Make it look nice
    fig.update_layout(
        title=f"Price Comparison: {ticker1} vs {ticker2} (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def calculate_volatility_metrics(df: DataFrame) -> DataFrame:
    # This helps us understand how much and how quickly prices change
    # Higher volatility means more risk but also more potential profit

    window_daily = Window.orderBy("Date").rowsBetween(-1, 0)

    # Calculate daily price changes and volatility metrics
    df_vol = df.withColumn(
        "daily_change",
        ((col("Close") - lag("Close", 1).over(Window.orderBy("Date"))) /
         lag("Close", 1).over(Window.orderBy("Date")) * 100)
    ).withColumn(
        "true_range",
        greatest(
            col("High") - col("Low"),  # Current day's range
            spark_abs(col("High") - lag("Close", 1).over(Window.orderBy("Date"))),  # Yesterday's close to today's high
            spark_abs(col("Low") - lag("Close", 1).over(Window.orderBy("Date")))  # Yesterday's close to today's low
        )
    )

    return df_vol


def explore_data(spark: SparkSession, ticker: str, days: int = 365):
    """Main function for data exploration."""
    st.subheader(f"Exploring data for {ticker}")

    # Fetch and prepare data
    df = get_stock_data(spark, ticker, days)

    # Stock Price History Visualization
    st.write("### Stock Price History")
    fig_price = plot_stock_price_history(df, ticker)
    st.plotly_chart(fig_price, use_container_width=True)

    # Display basic information
    st.write("### Data Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("First 40 rows:")
        st.dataframe(df.limit(40).toPandas(), height=400)
    with col2:
        st.write("Last 40 rows:")
        st.dataframe(df.orderBy(desc("Date")).limit(40).toPandas(), height=400)

    # Data frequency
    freq = analyze_data_frequency(df)
    st.write(f"### Data Frequency: {freq}")

    # Basic statistics
    st.write("### Basic Statistics")
    stats = calculate_basic_stats(df)
    st.dataframe(stats.toPandas())

    # Missing values analysis
    st.write("### Missing Values Analysis")
    missing_counts = df.select([sum(col(c).isNull().cast("int")).alias(c)
                                for c in df.columns])
    st.dataframe(missing_counts.toPandas())

    # Returns analysis
    st.write("### Returns Analysis")
    df_with_returns = calculate_returns(df)

    # Display returns visualization
    fig_returns = plot_returns_analysis(df_with_returns)
    st.plotly_chart(fig_returns, use_container_width=True)

    # Display summary statistics for returns
    st.write("### Returns Summary Statistics")
    returns_stats = df_with_returns.select(
        mean("daily_return").alias("Average Daily Return (%)"),
        stddev("daily_return").alias("Daily Return Volatility (%)"),
        mean("weekly_return").alias("Average Weekly Return (%)"),
        stddev("weekly_return").alias("Weekly Return Volatility (%)"),
        mean("monthly_return").alias("Average Monthly Return (%)"),
        stddev("monthly_return").alias("Monthly Return Volatility (%)")
    )
    st.dataframe(returns_stats.toPandas())

    # After returns analysis, add moving averages analysis
    st.write("### Moving Averages Analysis")

    # Calculate different moving averages for closing price
    ma_periods = [5, 20, 50]  # Common moving average periods
    df_with_ma = df_with_returns

    for period in ma_periods:
        df_with_ma = calculate_moving_average(df_with_ma, "Close", period)

    # Create moving averages visualization
    fig_ma = plot_moving_averages(df_with_ma, ticker, "Close", ma_periods)
    st.plotly_chart(fig_ma, use_container_width=True)

    # Display example calculation
    st.write("#### Moving Average Example Calculation")
    st.write("""
    Moving averages are calculated by taking the average of prices over a specified period.
    For example, a 5-day moving average is calculated as:
    ```
    MA = (Price₁ + Price₂ + Price₃ + Price₄ + Price₅) / 5
    ```
    """)

    # Show recent MA values
    st.write("Recent Moving Average Values:")
    recent_ma = df_with_ma.orderBy(desc("Date")).select(
        "Date",
        "Close",
        *[f"Close_MA_{period}" for period in ma_periods]
    ).limit(5)
    st.dataframe(recent_ma.toPandas())

    # Add trading signals analysis
    st.write("### Trading Signals Analysis")
    df_with_signals = analyze_ma_signals(df_with_ma, short_period=5, long_period=20)

    # Get current market position
    latest_signal = df_with_signals.orderBy(desc("Date")).select("Date", "signal").first()

    # Calculate current trend using multiple timeframes
    current_price = df_with_signals.orderBy(desc("Date")).select("Close").first().Close
    ma_5 = df_with_signals.orderBy(desc("Date")).select("Close_MA_5").first()["Close_MA_5"]
    ma_20 = df_with_signals.orderBy(desc("Date")).select("Close_MA_20").first()["Close_MA_20"]
    ma_50 = df_with_signals.orderBy(desc("Date")).select("Close_MA_50").first()["Close_MA_50"]

    # Create columns for timeframe analysis
    col1, col2, col3 = st.columns(3)

    # Multi-timeframe analysis
    with col1:
        st.write("#### Short-term Signal")
        if current_price > ma_5:
            st.success("Price above 5-day MA (Bullish)")  # Short-term uptrend
        else:
            st.error("Price below 5-day MA (Bearish)")  # Short-term downtrend

    with col2:
        st.write("#### Medium-term Signal")
        if current_price > ma_20:
            st.success("Price above 20-day MA (Bullish)")  # Medium-term uptrend
        else:
            st.error("Price below 20-day MA (Bearish)")  # Medium-term downtrend

    with col3:
        st.write("#### Long-term Signal")
        if current_price > ma_50:
            st.success("Price above 50-day MA (Bullish)")  # Long-term uptrend
        else:
            st.error("Price below 50-day MA (Bearish)")  # Long-term downtrend

    # Calculate overall sentiment
    recent_signals = df_with_signals.orderBy(desc("Date")).limit(10)
    bullish_count = recent_signals.filter(col("signal").contains("Bullish")).count()
    bearish_count = recent_signals.filter(col("signal").contains("Bearish")).count()

    # Sentiment strength calculation
    sentiment = "Bullish" if bullish_count > bearish_count else "Bearish"
    strength = abs(bullish_count - bearish_count) / 10  # Normalize to 0-1 scale

    # Generate insight message
    if latest_signal.signal == "Golden Cross (Buy)":
        st.success(f"🔔 Recent Golden Cross detected! This is typically a strong bullish signal.")
    elif latest_signal.signal == "Death Cross (Sell)":
        st.error(f"🔔 Recent Death Cross detected! This is typically a strong bearish signal.")

    # Overall market sentiment
    if sentiment == "Bullish":
        st.success(f"""
        ### Overall Market Sentiment: {sentiment} (Strength: {strength:.1%})
        
        Current Analysis for {ticker}:
        - Price is {'above' if current_price > ma_20 else 'below'} the 20-day moving average
        - Short-term trend (5-day MA) is {'upward' if ma_5 > ma_20 else 'downward'}
        - {'Strong buy signal' if strength > 0.7 else 'Moderate buy signal' if strength > 0.5 else 'Weak buy signal'}
        
        Always combine this technical analysis with fundamental analysis and your own research.
        """)
    else:
        st.error(f"""
        ### Overall Market Sentiment: {sentiment} (Strength: {strength:.1%})
        
        Current Analysis for {ticker}:
        - Price is {'above' if current_price > ma_20 else 'below'} the 20-day moving average
        - Short-term trend (5-day MA) is {'upward' if ma_5 > ma_20 else 'downward'}
        - {'Strong sell signal' if strength > 0.7 else 'Moderate sell signal' if strength > 0.5 else 'Weak sell signal'}
        
        Always combine this technical analysis with fundamental analysis and your own research.
        """)

    # Add stock correlation analysis
    st.write("### Stock Correlation Analysis")

    # Allow user to select another stock for comparison
    st.write("Compare with another stock:")
    comparison_method = st.radio(
        "Select comparison stock by:",
        ["Category", "Custom Ticker"],
        key="correlation_method"
    )

    if comparison_method == "Category":
        category = st.selectbox(
            "Select Category",
            list(STOCK_CATEGORIES.keys()),
            key="correlation_category"
        )
        stock_options = STOCK_CATEGORIES[category]
        comparison_ticker = st.selectbox(
            "Select Stock",
            [s for s in stock_options.keys() if s != ticker],
            format_func=lambda x: f"{x} - {stock_options[x]}",
            key="correlation_stock"
        )
    else:
        comparison_ticker = st.text_input(
            "Enter Stock Ticker for Comparison",
            max_chars=5,
            key="correlation_ticker"
        ).upper()

    if comparison_ticker and comparison_ticker != ticker:
        # Calculate correlations
        correlations = calculate_stock_correlation(spark, ticker, comparison_ticker, days)

        # Display correlation results
        st.write(f"#### Correlation between {ticker} and {comparison_ticker}")

        # Create correlation metrics display
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Price Correlation",
                      f"{correlations['Close']:.2f}",
                      help="Correlation between closing prices")

        with col2:
            st.metric("Volume Correlation",
                      f"{correlations['Volume']:.2f}",
                      help="Correlation between trading volumes")

        with col3:
            st.metric("Price Range Correlation",
                      f"{(correlations['High'] + correlations['Low']) / 2:.2f}",
                      help="Average correlation of price ranges")

        # Get comparison stock data
        df_comparison = get_stock_data(spark, comparison_ticker, days)

        # Create comparison visualization
        fig_comparison = plot_correlation_comparison(df, df_comparison, ticker, comparison_ticker)
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Add correlation interpretation
        st.write("#### Correlation Interpretation")
        correlation_value = correlations['Close']
        if abs(correlation_value) > 0.7:
            st.write(f"Strong {'positive' if correlation_value > 0 else 'negative'} correlation")
        elif abs(correlation_value) > 0.3:
            st.write(f"Moderate {'positive' if correlation_value > 0 else 'negative'} correlation")
        else:
            st.write("Weak correlation")

        st.write("""
        Correlation ranges from -1 to +1:
        - +1: Perfect positive correlation
        - 0: No correlation
        - -1: Perfect negative correlation
        
        Note: Correlation doesn't imply causation. Always consider other factors in your analysis.
        """)

    # Add 8 Key Insights Section
    st.write("### 🔍 Key Market Insights")

    # 1. Volume-Price Correlation Analysis
    st.write("#### 1. Volume-Price Relationship")
    volume_price_corr = df.select(corr("Volume", "Close")).first()[0]
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Volume-Price Correlation",
            f"{volume_price_corr:.2f}",
            help="Correlation between trading volume and closing price"
        )

    # 2. Trading Range Analysis
    df_stats = df.agg(
        mean("High").alias("avg_high"),
        mean("Low").alias("avg_low"),
        mean("Close").alias("avg_close")
    ).first()

    with col2:
        trading_range = ((df_stats["avg_high"] - df_stats["avg_low"]) / df_stats["avg_close"]) * 100
        st.metric(
            "Average Trading Range",
            f"{trading_range:.2f}%",
            help="Average price range as percentage of closing price"
        )

    # 3. Volume Profile
    st.write("#### 2. Volume Analysis")
    df_vol = calculate_volatility_metrics(df)
    avg_volume = df.select(mean("Volume")).first()[0]
    high_volume_days = df.filter(col("Volume") > avg_volume * 1.5).count()
    volume_ratio = (high_volume_days / df.count()) * 100

    st.write(f"- {volume_ratio:.1f}% of trading days show high volume (>50% above average)")

    # 4. Price Momentum
    st.write("#### 3. Price Momentum")
    momentum_window = 10
    df_momentum = df_vol.withColumn(
        "momentum_10d",
        ((col("Close") - lag("Close", momentum_window).over(Window.orderBy("Date"))) /
         lag("Close", momentum_window).over(Window.orderBy("Date")) * 100)
    )

    recent_momentum = df_momentum.orderBy(desc("Date")).select("momentum_10d").first()[0]
    st.metric(
        "10-Day Price Momentum",
        f"{recent_momentum:.2f}%",
        help="Percentage price change over last 10 trading days"
    )

    # 5. Volatility Analysis
    st.write("#### 4. Volatility Analysis")
    daily_volatility = df_vol.select(stddev("daily_change")).first()[0]
    avg_true_range = df_vol.select(mean("true_range")).first()[0]

    col3, col4 = st.columns(2)
    with col3:
        st.metric(
            "Daily Volatility",
            f"{daily_volatility:.2f}%",
            help="Standard deviation of daily price changes"
        )
    with col4:
        st.metric(
            "Average True Range",
            f"${avg_true_range:.2f}",
            help="Average of true price ranges"
        )

    # 6. Gap Analysis
    st.write("#### 5. Price Gap Analysis")
    df_gaps = df.withColumn(
        "gap",
        ((col("Open") - lag("Close", 1).over(Window.orderBy("Date"))) /
         lag("Close", 1).over(Window.orderBy("Date")) * 100)
    )

    significant_gaps = df_gaps.filter(spark_abs(col("gap")) > 1).count()
    gap_ratio = (significant_gaps / df.count()) * 100
    st.write(f"- {gap_ratio:.1f}% of trading days show significant gaps (>1%)")

    # 7. Support/Resistance Levels
    st.write("#### 6. Support & Resistance Levels")
    price_distribution = df.select(
        percentile_approx("Close", array(lit(0.1), lit(0.9))).alias("price_levels")
    ).first()

    support_level = price_distribution["price_levels"][0]
    resistance_level = price_distribution["price_levels"][1]
    current_price = df.orderBy(desc("Date")).select("Close").first()[0]

    col5, col6 = st.columns(2)
    with col5:
        st.metric(
            "Support Level",
            f"${support_level:.2f}",
            delta=f"{((current_price - support_level) / support_level * 100):.1f}% from current"
        )
    with col6:
        st.metric(
            "Resistance Level",
            f"${resistance_level:.2f}",
            delta=f"{((resistance_level - current_price) / current_price * 100):.1f}% from current"
        )

    # 8. Market Efficiency
    st.write("#### 7. Market Efficiency Metrics")
    # Calculate how often price moves follow the previous day's direction
    df_efficiency = df_vol.withColumn(
        "trend_following",
        when(
            ((col("daily_change") > 0) & (lag("daily_change", 1).over(Window.orderBy("Date")) > 0)) |
            ((col("daily_change") < 0) & (lag("daily_change", 1).over(Window.orderBy("Date")) < 0)),
            1
        ).otherwise(0)
    )

    efficiency_ratio = df_efficiency.select(
        coalesce(mean("trend_following"), lit(0))
    ).first()[0] * 100

    st.metric(
        "Market Efficiency Ratio",
        f"{efficiency_ratio:.1f}%",
        help="Percentage of price moves following previous day's direction"
    )

    # Summary Insights
    st.write("#### 8. Summary Analysis")
    current_metrics = df.orderBy(desc("Date")).first()
    avg_metrics = df.agg(
        mean("Close").alias("avg_price"),
        mean("Volume").alias("avg_volume")
    ).first()

    if current_metrics and avg_metrics:
        price_vs_avg = ((current_metrics["Close"] - avg_metrics["avg_price"]) /
                        avg_metrics["avg_price"] * 100)
        volume_vs_avg = ((current_metrics["Volume"] - avg_metrics["avg_volume"]) /
                         avg_metrics["avg_volume"] * 100)

        st.write(f"""
        Current Market Position:
        - Price is {abs(float(price_vs_avg)):.1f}% {'above' if price_vs_avg > 0 else 'below'} the period average
        - Volume is {abs(float(volume_vs_avg)):.1f}% {'above' if volume_vs_avg > 0 else 'below'} the period average
        - Volatility is {'high' if daily_volatility > 2 else 'moderate' if daily_volatility > 1 else 'low'}
        - Market efficiency suggests {'trending' if efficiency_ratio > 60 else 'ranging'} behavior
        """)
    else:
        st.warning("Insufficient data for summary analysis")

    return df_with_signals
