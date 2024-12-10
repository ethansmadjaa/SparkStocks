from pyspark.sql import DataFrame, SparkSession
import streamlit as st
from pyspark.sql.functions import (
    col, date_trunc, desc, first, lag, 
    mean, stddev, min, max, datediff, sum
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
    window_spec = Window.orderBy("Date")\
                       .partitionBy(date_trunc("month", col("Date")))
    
    df_with_diff = df.withColumn("date_diff", 
        datediff(col("Date"), lag("Date", 1).over(window_spec)))
    
    mode_freq = df_with_diff.groupBy("date_diff")\
                           .count()\
                           .orderBy(desc("count"))\
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
    window_week = Window.orderBy("Date")\
                       .rowsBetween(-5, 0)\
                       .partitionBy(date_trunc("week", col("Date")))
    
    df = df.withColumn("weekly_return", 
        ((col("Close") - first("Close").over(window_week)) / 
         first("Close").over(window_week) * 100))
    
    # Monthly returns (using 21 trading days)
    window_month = Window.orderBy("Date")\
                        .rowsBetween(-21, 0)\
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
    
    return df_with_returns
