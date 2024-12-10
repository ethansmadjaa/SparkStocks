from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.window import Window

def calculate_rsi(df: DataFrame, period: int = 14) -> DataFrame:
    """Calculate Relative Strength Index."""
    window_spec = Window.orderBy("Date")
    
    # Calculate price changes
    df = df.withColumn("price_change", col("Close") - lag("Close", 1).over(window_spec))
    
    # Calculate gains (positive changes) and losses (negative changes)
    df = df.withColumn(
        "gain", when(col("price_change") > 0, col("price_change")).otherwise(0)
    ).withColumn(
        "loss", when(col("price_change") < 0, abs(col("price_change"))).otherwise(0)
    )
    
    # Calculate average gains and losses
    df = df.withColumn(
        "avg_gain", avg("gain").over(Window.orderBy("Date").rowsBetween(0, period-1))
    ).withColumn(
        "avg_loss", avg("loss").over(Window.orderBy("Date").rowsBetween(0, period-1))
    )
    
    # Calculate RSI
    df = df.withColumn(
        "rs", col("avg_gain") / col("avg_loss")
    ).withColumn(
        "rsi", 100 - (100 / (1 + col("rs")))
    )
    
    return df

def calculate_macd(df: DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> DataFrame:
    """Calculate Moving Average Convergence Divergence."""
    window_spec = Window.orderBy("Date")
    
    # Calculate EMAs
    df = df.withColumn(
        "ema_fast", 
        exp(avg(ln("Close")).over(Window.orderBy("Date").rowsBetween(-fast_period, 0)))
    ).withColumn(
        "ema_slow", 
        exp(avg(ln("Close")).over(Window.orderBy("Date").rowsBetween(-slow_period, 0)))
    )
    
    # Calculate MACD line
    df = df.withColumn("macd_line", col("ema_fast") - col("ema_slow"))
    
    # Calculate signal line
    df = df.withColumn(
        "signal_line",
        exp(avg(ln("macd_line")).over(Window.orderBy("Date").rowsBetween(-signal_period, 0)))
    )
    
    # Calculate histogram
    df = df.withColumn("macd_histogram", col("macd_line") - col("signal_line"))
    
    return df

def calculate_trading_signals(df: DataFrame) -> DataFrame:
    """Calculate trading signals based on technical indicators."""
    # Ensure we have all required indicators
    df = calculate_rsi(df)
    df = calculate_macd(df)
    
    # Generate trading signals
    df = df.withColumn(
        "signal",
        when((col("rsi") < 30) & (col("macd_line") > col("signal_line")), "Strong Buy")
        .when((col("rsi") < 40) & (col("macd_line") > col("signal_line")), "Buy")
        .when((col("rsi") > 70) & (col("macd_line") < col("signal_line")), "Strong Sell")
        .when((col("rsi") > 60) & (col("macd_line") < col("signal_line")), "Sell")
        .otherwise("Hold")
    )
    
    return df