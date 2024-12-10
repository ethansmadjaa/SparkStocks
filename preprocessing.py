from pyspark.sql import DataFrame, SparkSession
import streamlit as st
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from exploration import get_stock_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_rsi(df: DataFrame, period: int = 14) -> DataFrame:
    """Calculate Relative Strength Index."""
    # Calculate price changes
    window_spec = Window.orderBy("Date")
    df = df.withColumn("price_change", 
                      col("Close") - lag("Close", 1).over(window_spec))
    
    # Calculate gains (positive changes) and losses (negative changes)
    df = df.withColumn("gain", when(col("price_change") > 0, col("price_change")).otherwise(0))
    df = df.withColumn("loss", when(col("price_change") < 0, -col("price_change")).otherwise(0))
    
    # Calculate average gains and losses
    window_avg = Window.orderBy("Date").rowsBetween(-period, 0)
    df = df.withColumn("avg_gain", avg("gain").over(window_avg))
    df = df.withColumn("avg_loss", avg("loss").over(window_avg))
    
    # Calculate RSI
    df = df.withColumn("rs", col("avg_gain") / col("avg_loss"))
    df = df.withColumn("rsi", 100 - (100 / (1 + col("rs"))))
    
    return df

def calculate_macd(df: DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    # Calculate EMAs
    df = df.withColumn(
        f"ema_{fast_period}",
        exp(avg(log("Close")).over(Window.orderBy("Date").rowsBetween(-fast_period, 0)))
    )
    
    df = df.withColumn(
        f"ema_{slow_period}",
        exp(avg(log("Close")).over(Window.orderBy("Date").rowsBetween(-slow_period, 0)))
    )
    
    # Calculate MACD line
    df = df.withColumn("macd_line", 
                      col(f"ema_{fast_period}") - col(f"ema_{slow_period}"))
    
    # Calculate Signal line
    df = df.withColumn(
        "signal_line",
        exp(avg(log("macd_line")).over(Window.orderBy("Date").rowsBetween(-signal_period, 0)))
    )
    
    # Calculate MACD histogram
    df = df.withColumn("macd_histogram", 
                      col("macd_line") - col("signal_line"))
    
    return df

def calculate_bollinger_bands(df: DataFrame, period: int = 20, std_dev: float = 2.0) -> DataFrame:
    """Calculate Bollinger Bands."""
    window_spec = Window.orderBy("Date").rowsBetween(-period, 0)
    
    # Calculate middle band (20-day SMA)
    df = df.withColumn("bb_middle", avg("Close").over(window_spec))
    
    # Calculate standard deviation
    df = df.withColumn("bb_std", 
                      stddev("Close").over(window_spec))
    
    # Calculate upper and lower bands
    df = df.withColumn("bb_upper", 
                      col("bb_middle") + (col("bb_std") * std_dev))
    df = df.withColumn("bb_lower", 
                      col("bb_middle") - (col("bb_std") * std_dev))
    
    return df

def plot_technical_indicators(df: DataFrame, ticker: str):
    """Create visualization for technical indicators."""
    pdf = df.toPandas()
    
    fig = make_subplots(rows=4, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('Price with Bollinger Bands', 'RSI', 'MACD', 'Volume'),
                       row_heights=[0.4, 0.2, 0.2, 0.2])
    
    # Price and Bollinger Bands
    fig.add_trace(
        go.Scatter(x=pdf['Date'], y=pdf['Close'], name='Close Price',
                  line=dict(color='blue')), row=1, col=1)
    
    fig.add_trace(
        go.Scatter(x=pdf['Date'], y=pdf['bb_upper'], name='Upper BB',
                  line=dict(color='gray', dash='dash')), row=1, col=1)
    
    fig.add_trace(
        go.Scatter(x=pdf['Date'], y=pdf['bb_lower'], name='Lower BB',
                  line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # RSI
    fig.add_trace(
        go.Scatter(x=pdf['Date'], y=pdf['rsi'], name='RSI',
                  line=dict(color='purple')), row=2, col=1)
    
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=pdf['Date'], y=pdf['macd_line'], name='MACD',
                  line=dict(color='blue')), row=3, col=1)
    
    fig.add_trace(
        go.Scatter(x=pdf['Date'], y=pdf['signal_line'], name='Signal',
                  line=dict(color='orange')), row=3, col=1)
    
    fig.add_trace(
        go.Bar(x=pdf['Date'], y=pdf['macd_histogram'], name='MACD Histogram',
               marker_color='gray'), row=3, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(x=pdf['Date'], y=pdf['Volume'], name='Volume',
               marker_color='lightblue'), row=4, col=1)
    
    fig.update_layout(
        height=1000,
        title_text=f"Technical Analysis Dashboard - {ticker}",
        showlegend=True
    )
    
    return fig

def preprocess_data(spark: SparkSession, ticker: str, days: int = 365):
    """Main function for preprocessing."""
    st.subheader(f"📊 Advanced Technical Analysis for {ticker}")
    
    # Get stock data
    df = get_stock_data(spark, ticker, days)
    
    # Calculate technical indicators
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    
    # Display technical analysis dashboard
    fig = plot_technical_indicators(df, ticker)
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical Indicators Interpretation
    st.write("### Technical Indicators Analysis")
    
    # Get latest values
    latest = df.orderBy(desc("Date")).first()
    
    # RSI Analysis
    st.write("#### RSI Analysis")
    rsi_value = latest["rsi"]
    rsi_signal = ("Oversold" if rsi_value < 30 else 
                 "Overbought" if rsi_value > 70 else 
                 "Neutral")
    
    st.metric(
        "RSI (14)",
        f"{rsi_value:.2f}",
        delta=rsi_signal,
        delta_color="off"
    )
    
    # MACD Analysis
    st.write("#### MACD Analysis")
    macd_signal = "Bullish" if latest["macd_line"] > latest["signal_line"] else "Bearish"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "MACD Line",
            f"{latest['macd_line']:.2f}",
            delta=macd_signal,
            delta_color="off"
        )
    with col2:
        st.metric(
            "Signal Line",
            f"{latest['signal_line']:.2f}"
        )
    
    # Bollinger Bands Analysis
    st.write("#### Bollinger Bands Analysis")
    bb_position = ((latest["Close"] - latest["bb_lower"]) / 
                  (latest["bb_upper"] - latest["bb_lower"]) * 100)
    
    bb_signal = ("Oversold" if bb_position < 20 else 
                "Overbought" if bb_position > 80 else 
                "Neutral")
    
    st.metric(
        "Price Position in BB (%)",
        f"{bb_position:.2f}%",
        delta=bb_signal,
        delta_color="off"
    )
    
    # Export options
    st.write("### Export Processed Data")
    if st.button("Download Technical Indicators Data"):
        # Convert to pandas and prepare for download
        export_df = df.select(
            "Date", "Close", "Volume",
            "rsi", "macd_line", "signal_line", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower"
        ).toPandas()
        
        st.download_button(
            label="Download CSV",
            data=export_df.to_csv(index=False).encode('utf-8'),
            file_name=f"{ticker}_technical_indicators.csv",
            mime="text/csv"
        )
    
    return df
