from pyspark.sql import DataFrame, SparkSession
import streamlit as st
from pyspark.sql.functions import (
    col, date_trunc, desc, first, lag,
    mean, stddev, min, max, datediff, sum, when
)
from pyspark.sql.window import Window
from pyspark.sql.types import *
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


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
    """Analyze the frequency of data points with optimized window operation."""
    window_spec = Window.orderBy("Date") \
        .partitionBy(date_trunc("month", col("Date")))

    df_with_diff = df.withColumn("date_diff",
                                 datediff(col("Date"), lag("Date", 1).over(window_spec)))

    mode_freq = df_with_diff.groupBy("date_diff") \
        .count() \
        .orderBy(desc("count")) \
        .first()

    freq_mapping = {
        1: "Daily",
        7: "Weekly",
        30: "Monthly",
        365: "Yearly"
    }
    return freq_mapping.get(mode_freq["date_diff"], f"Custom ({mode_freq['date_diff']} days)")


def calculate_basic_stats(df: DataFrame) -> DataFrame:
    """Calculate basic statistics for numerical columns."""
    numeric_cols = [f.name for f in df.schema.fields
                    if isinstance(f.dataType, (DoubleType, LongType))]

    # Create a list of expressions for the select statement
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
    """Calculate daily, weekly, monthly returns with optimized window operations."""
    # Partition by date for better performance
    df = df.repartition("Date")

    # Daily returns
    df = df.withColumn("daily_return",
                       (col("Close") - col("Open")) / col("Open") * 100)

    # Weekly returns (using 5 trading days)
    window_week = Window.orderBy("Date") \
        .rowsBetween(-5, 0) \
        .partitionBy(date_trunc("week", col("Date")))

    df = df.withColumn("weekly_return",
                       ((col("Close") - first("Close").over(window_week)) /
                        first("Close").over(window_week) * 100))

    # Monthly returns (using 21 trading days)
    window_month = Window.orderBy("Date") \
        .rowsBetween(-21, 0) \
        .partitionBy(date_trunc("month", col("Date")))

    df = df.withColumn("monthly_return",
                       ((col("Close") - first("Close").over(window_month)) /
                        first("Close").over(window_month) * 100))

    return df


def plot_stock_price_history(df: DataFrame, ticker: str):
    """Create an interactive candlestick chart with volume."""
    # Convert to pandas for plotting
    pdf = df.toPandas()

    # Create the candlestick chart with volume subplot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=(f'{ticker} Stock Price', 'Volume'),
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=pdf['Date'],
                                 open=pdf['Open'],
                                 high=pdf['High'],
                                 low=pdf['Low'],
                                 close=pdf['Close'],
                                 name='OHLC'),
                  row=1, col=1)

    fig.add_trace(go.Bar(x=pdf['Date'],
                         y=pdf['Volume'],
                         name='Volume'),
                  row=2, col=1)

    fig.update_layout(
        title=f'{ticker} Stock Price History',
        yaxis_title='Stock Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=800
    )

    return fig


def plot_returns_analysis(df: DataFrame):
    """Create comprehensive returns analysis visualizations."""
    pdf = df.select('Date', 'daily_return', 'weekly_return', 'monthly_return').toPandas()

    # Create subplots: Returns over time and Box plots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Returns Over Time', 'Returns Distribution (Box Plot)'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )

    # Returns over time
    fig.add_trace(
        go.Scatter(x=pdf['Date'], y=pdf['daily_return'],
                   name='Daily Returns', mode='lines',
                   line=dict(color='blue', width=1)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=pdf['Date'], y=pdf['weekly_return'],
                   name='Weekly Returns', mode='lines',
                   line=dict(color='green', width=1.5)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=pdf['Date'], y=pdf['monthly_return'],
                   name='Monthly Returns', mode='lines',
                   line=dict(color='red', width=2)),
        row=1, col=1
    )

    # Box plots for returns distribution
    fig.add_trace(
        go.Box(y=pdf['daily_return'], name='Daily',
               boxpoints='outliers', jitter=0.3,
               whiskerwidth=0.2, marker_color='blue',
               line_width=1),
        row=2, col=1
    )

    fig.add_trace(
        go.Box(y=pdf['weekly_return'], name='Weekly',
               boxpoints='outliers', jitter=0.3,
               whiskerwidth=0.2, marker_color='green',
               line_width=1),
        row=2, col=1
    )

    fig.add_trace(
        go.Box(y=pdf['monthly_return'], name='Monthly',
               boxpoints='outliers', jitter=0.3,
               whiskerwidth=0.2, marker_color='red',
               line_width=1),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes
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
    """
    Analyze moving average crossovers to generate trading signals.
    
    Trading Signal Logic:
    1. Golden Cross (Strong Buy): When short MA crosses above long MA
    2. Death Cross (Strong Sell): When short MA crosses below long MA
    3. Bullish: When short MA is above long MA
    4. Bearish: When short MA is below long MA
    
    Args:
        df: DataFrame with price data and MAs
        short_period: Period for shorter MA (default: 5 days)
        long_period: Period for longer MA (default: 20 days)
    """
    # Calculate the difference between short and long MA
    signal_col = f"Close_MA_{short_period}_vs_{long_period}"
    df = df.withColumn(signal_col,
                       col(f"Close_MA_{short_period}") - col(f"Close_MA_{long_period}"))

    # Create a window spec for looking at previous value
    window_spec = Window.orderBy("Date").rowsBetween(-1, 0)

    # Detect crossovers by comparing current and previous signals
    df = df.withColumn("prev_signal",
                       lag(signal_col, 1).over(Window.orderBy("Date")))

    # Generate trading signals based on crossovers
    df = df.withColumn("signal",
                       when((col(signal_col) > 0) & (col("prev_signal") < 0),
                            "Golden Cross (Buy)")  # Short MA crosses above Long MA
                       .when((col(signal_col) < 0) & (col("prev_signal") > 0),
                             "Death Cross (Sell)")  # Short MA crosses below Long MA
                       .when(col(signal_col) > 0, "Bullish")  # Short MA above Long MA
                       .when(col(signal_col) < 0, "Bearish")  # Short MA below Long MA
                       .otherwise("Neutral"))

    return df


def explore_data(spark: SparkSession, ticker: str, days: int = 365):
    """Main function for data exploration."""
    st.subheader(f"📊 Exploring data for {ticker}")

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

    return df_with_signals
