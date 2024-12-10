# This is where we do all the technical analysis stuff
# We calculate things like RSI, MACD, and Bollinger Bands to help understand stock movements
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

from utils.data_utils import get_stock_data


def calculate_rsi(df: DataFrame, period: int = 14) -> DataFrame:
    # RSI (Relative Strength Index) tells us if a stock is overbought or oversold
    # It looks at the last 14 days by default and gives us a number between 0 and 100
    # Above 70 means probably overbought (might go down soon)
    # Below 30 means probably oversold (might go up soon)
    
    # Add partitioning by year and month for better performance
    df = df.withColumn("year", year("Date"))
    df = df.withColumn("month", month("Date"))
    
    # Define window spec with partitioning
    window_spec = Window.partitionBy("year", "month").orderBy("Date")
    
    df = df.withColumn("price_change", 
                      col("Close") - lag("Close", 1).over(window_spec))
    
    # Split these changes into gains and losses
    # If price went up, it's a gain. If it went down, it's a loss
    df = df.withColumn("gain", when(col("price_change") > 0, col("price_change")).otherwise(0))
    df = df.withColumn("loss", when(col("price_change") < 0, -col("price_change")).otherwise(0))
    
    # Now we average these gains and losses over our period (usually 14 days)
    window_avg = Window.partitionBy("year", "month") \
                      .orderBy("Date") \
                      .rowsBetween(-period, 0)
    df = df.withColumn("avg_gain", avg("gain").over(window_avg))
    df = df.withColumn("avg_loss", avg("loss").over(window_avg))
    
    # Finally, calculate RSI using the formula: 100 - (100 / (1 + RS))
    # where RS = average gain / average loss
    df = df.withColumn("rs", col("avg_gain") / col("avg_loss"))
    df = df.withColumn("rsi", 100 - (100 / (1 + col("rs"))))
    
    # Drop the partitioning columns at the end
    return df.drop("year", "month")


def calculate_macd(df: DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> DataFrame:
    # MACD helps us spot changes in the trend and momentum of a stock
    # It uses three periods: fast (12 days), slow (26 days), and signal (9 days)
    # When the MACD line crosses above the signal line, it might be time to buy
    # When it crosses below, it might be time to sell
    
    # Add partitioning like we did for RSI
    df = df.withColumn("year", year("Date"))
    df = df.withColumn("month", month("Date"))
    
    window_spec = Window.partitionBy("year", "month").orderBy("Date")
    
    # Calculate EMAs with null handling
    df = df.withColumn(
        f"ema_{fast_period}",
        coalesce(
            exp(avg(log("Close")).over(window_spec.rowsBetween(-fast_period, 0))),
            col("Close")
        )
    )
    
    df = df.withColumn(
        f"ema_{slow_period}",
        coalesce(
            exp(avg(log("Close")).over(window_spec.rowsBetween(-slow_period, 0))),
            col("Close")
        )
    )
    
    # MACD line with null handling
    df = df.withColumn("macd_line", 
                      coalesce(
                          col(f"ema_{fast_period}") - col(f"ema_{slow_period}"),
                          lit(0.0)
                      ))
    
    # Signal line with null handling
    df = df.withColumn(
        "signal_line",
        coalesce(
            exp(avg(log(abs(col("macd_line")) + 0.00001)).over(window_spec.rowsBetween(-signal_period, 0))),
            col("macd_line")
        )
    )
    
    # Histogram with null handling
    df = df.withColumn("macd_histogram", 
                      coalesce(
                          col("macd_line") - col("signal_line"),
                          lit(0.0)
                      ))
    
    # Drop the partitioning columns
    return df.drop("year", "month")


def calculate_bollinger_bands(df: DataFrame, period: int = 20, std_dev: float = 2.0) -> DataFrame:
    # Bollinger Bands show us the volatility and potential price levels to watch
    # They're like a price channel that gets wider when volatility is high
    # and narrower when volatility is low
    
    # We look at a 20-day period by default
    window_spec = Window.orderBy("Date").rowsBetween(-period, 0)
    
    # Middle band is just a simple moving average
    df = df.withColumn("bb_middle", avg("Close").over(window_spec))
    
    # Calculate how much prices typically deviate from the average
    df = df.withColumn("bb_std", 
                      stddev("Close").over(window_spec))
    
    # Upper and lower bands are 2 standard deviations from the middle
    # This means about 95% of prices should fall between these bands
    df = df.withColumn("bb_upper", 
                      col("bb_middle") + (col("bb_std") * std_dev))
    df = df.withColumn("bb_lower", 
                      col("bb_middle") - (col("bb_std") * std_dev))
    
    return df


def calculate_trading_signals(df: DataFrame) -> DataFrame:
    """Calculate trading signals and market sentiment"""
    
    # Add partitioning for better performance
    df = df.withColumn("year", year("Date"))
    df = df.withColumn("month", month("Date"))
    window_spec = Window.partitionBy("year", "month").orderBy("Date")
    
    # Calculate moving averages for different periods
    for period in [5, 10, 20, 50, 200]:
        df = df.withColumn(
            f"ma_{period}",
            avg("Close").over(window_spec.rowsBetween(-period, 0))
        )
    
    # Calculate signals with weighted importance
    df = df.withColumn(
        "signal_short_term",
        when(col("Close") > col("ma_10"), 2)  # More weight to short-term
        .when(col("Close") < col("ma_10"), -2)
        .otherwise(0)
    )
    
    df = df.withColumn(
        "signal_medium_term",
        when(col("Close") > col("ma_50"), 1.5)
        .when(col("Close") < col("ma_50"), -1.5)
        .otherwise(0)
    )
    
    df = df.withColumn(
        "signal_long_term",
        when(col("Close") > col("ma_200"), 1)
        .when(col("Close") < col("ma_200"), -1)
        .otherwise(0)
    )
    
    # Calculate momentum
    df = df.withColumn(
        "price_momentum",
        when(col("Close") > lag("Close", 1).over(window_spec), 1)
        .when(col("Close") < lag("Close", 1).over(window_spec), -1)
        .otherwise(0)
    )
    
    # Calculate RSI signal with more granular levels
    df = df.withColumn(
        "signal_rsi",
        when(col("rsi") < 30, 2)  # Strong oversold
        .when(col("rsi") < 40, 1)  # Moderately oversold
        .when(col("rsi") > 70, -2)  # Strong overbought
        .when(col("rsi") > 60, -1)  # Moderately overbought
        .otherwise(0)
    )
    
    # Calculate MACD signal with trend strength
    df = df.withColumn(
        "signal_macd",
        when((col("macd_line") > col("signal_line")) & 
             (col("macd_histogram") > lag("macd_histogram", 1).over(window_spec)), 2)  # Strong bullish
        .when(col("macd_line") > col("signal_line"), 1)  # Bullish
        .when((col("macd_line") < col("signal_line")) & 
              (col("macd_histogram") < lag("macd_histogram", 1).over(window_spec)), -2)  # Strong bearish
        .when(col("macd_line") < col("signal_line"), -1)  # Bearish
        .otherwise(0)
    )
    
    # Volume trend
    df = df.withColumn(
        "volume_trend",
        when((col("Volume") > lag("Volume", 1).over(window_spec)) & 
             (col("Close") > lag("Close", 1).over(window_spec)), 1)  # Up volume
        .when((col("Volume") > lag("Volume", 1).over(window_spec)) & 
              (col("Close") < lag("Close", 1).over(window_spec)), -1)  # Down volume
        .otherwise(0)
    )
    
    # Calculate overall sentiment with weighted signals
    df = df.withColumn(
        "total_signals",
        col("signal_short_term") +  # Weight: 2
        col("signal_medium_term") +  # Weight: 1.5
        col("signal_long_term") +    # Weight: 1
        col("signal_rsi") +          # Weight: 2
        col("signal_macd") +         # Weight: 2
        col("price_momentum") +      # Weight: 1
        col("volume_trend")          # Weight: 1
    )
    
    # Calculate sentiment strength (normalize between 0 and 1)
    max_possible_score = 10.5  # Sum of all positive weights
    df = df.withColumn(
        "sentiment_strength",
        (col("total_signals") + max_possible_score) / (2 * max_possible_score)  # Normalize to [0,1]
    )
    
    # More nuanced sentiment classification
    df = df.withColumn(
        "sentiment",
        when(col("total_signals") > 3, "Strongly Bullish")
        .when(col("total_signals") > 0, "Bullish")
        .when(col("total_signals") < -3, "Strongly Bearish")
        .when(col("total_signals") < 0, "Bearish")
        .otherwise("Neutral")
    )
    
    # Drop partitioning columns and temporary columns
    return df.drop("year", "month", "price_momentum", "volume_trend")


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
    
    # Get stock data and calculate all indicators
    df = get_stock_data(spark, ticker, days)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_trading_signals(df)
    
    # Display technical analysis dashboard
    fig = plot_technical_indicators(df, ticker)
    st.plotly_chart(fig, use_container_width=True)
    
    # Get latest values for analysis
    latest = df.orderBy(desc("Date")).first()
    
    # Trading Signals Analysis
    st.write("### Trading Signals Analysis")
    
    sentiment = latest["sentiment"]
    strength = latest["sentiment_strength"]
    
    # Create a more detailed analysis message
    analysis_message = f"""
    #### Overall Market Sentiment: {sentiment} (Strength: {strength:.1%})
    
    Current Signals for {ticker}:
    - Short-term trend (10-day MA): {"Bullish" if latest["signal_short_term"] > 0 else "Bearish"}
    - Medium-term trend (50-day MA): {"Bullish" if latest["signal_medium_term"] > 0 else "Bearish"}
    - Long-term trend (200-day MA): {"Bullish" if latest["signal_long_term"] > 0 else "Bearish"}
    - RSI Signal: {"Oversold (Bullish)" if latest["signal_rsi"] > 0 else "Overbought (Bearish)" if latest["signal_rsi"] < 0 else "Neutral"}
    - MACD Signal: {"Bullish" if latest["signal_macd"] > 0 else "Bearish"}
    
    Price is currently:
    - {"Above" if latest["Close"] > latest["ma_200"] else "Below"} 200-day moving average
    - {"Above" if latest["Close"] > latest["ma_50"] else "Below"} 50-day moving average
    - {"Above" if latest["Close"] > latest["ma_10"] else "Below"} 10-day moving average
    
    Always combine this technical analysis with fundamental research before making investment decisions.
    """
    
    if sentiment == "Bullish":
        st.success(analysis_message)
    elif sentiment == "Bearish":
        st.error(analysis_message)
    else:
        st.info(analysis_message)
    
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
    macd_value = float(latest["macd_line"] or 0.0)
    signal_value = float(latest["signal_line"] or 0.0)
    macd_signal = "Bullish" if macd_value > signal_value else "Bearish"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "MACD Line",
            f"{macd_value:.2f}",
            delta=macd_signal,
            delta_color="off"
        )
    with col2:
        st.metric(
            "Signal Line",
            f"{signal_value:.2f}"
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
