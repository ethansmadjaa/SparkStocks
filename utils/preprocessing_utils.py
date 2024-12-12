from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from typing import Optional

def calculate_rsi(df: DataFrame, period: int = 14, cleanup: bool = True) -> DataFrame:
    """
    Calculate Relative Strength Index with memory optimization.
    
    Args:
        df: Input DataFrame
        period: RSI period
        cleanup: Whether to drop intermediate columns
    """
    # Add partitioning by year and month for better performance
    df = df.withColumn("year", year("Date")).withColumn("month", month("Date"))
    
    # Window for lag calculation (no frame specification needed for lag)
    lag_window = Window.partitionBy("year", "month").orderBy("Date")
    
    # Window for averages calculation
    avg_window = Window.partitionBy("year", "month").orderBy("Date").rowsBetween(-period + 1, 0)
    
    # Calculate price changes
    df = df.withColumn("price_change", col("Close") - lag("Close", 1).over(lag_window))
    
    # Calculate gains and losses
    df = df.withColumn(
        "gain", when(col("price_change") > 0, col("price_change")).otherwise(0)
    ).withColumn(
        "loss", when(col("price_change") < 0, abs(col("price_change"))).otherwise(0)
    )
    
    # Calculate RSI
    df = df.withColumn(
        "avg_gain", avg("gain").over(avg_window)
    ).withColumn(
        "avg_loss", avg("loss").over(avg_window)
    ).withColumn(
        "rs", col("avg_gain") / col("avg_loss")
    ).withColumn(
        "rsi", 100 - (100 / (1 + col("rs")))
    )
    
    # Cleanup intermediate columns
    if cleanup:
        df = df.drop("price_change", "gain", "loss", "avg_gain", "avg_loss", "rs", "year", "month")
    
    return df

def calculate_macd(
    df: DataFrame, 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9,
    cleanup: bool = True
) -> DataFrame:
    """
    Calculate MACD with memory optimization.
    
    Args:
        df: Input DataFrame
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        cleanup: Whether to drop intermediate columns
    """
    # Add partitioning by year and month
    df = df.withColumn("year", year("Date")).withColumn("month", month("Date"))
    
    # Define windows for different periods
    window_fast = Window.partitionBy("year", "month").orderBy("Date").rowsBetween(-fast_period + 1, 0)
    window_slow = Window.partitionBy("year", "month").orderBy("Date").rowsBetween(-slow_period + 1, 0)
    window_signal = Window.partitionBy("year", "month").orderBy("Date").rowsBetween(-signal_period + 1, 0)
    
    # Calculate EMAs
    df = df.withColumn(
        "ema_fast", 
        exp(avg(ln("Close")).over(window_fast))
    ).withColumn(
        "ema_slow", 
        exp(avg(ln("Close")).over(window_slow))
    )
    
    # Calculate MACD line
    df = df.withColumn(
        "macd_line", 
        col("ema_fast") - col("ema_slow")
    )
    
    # Calculate signal line and histogram
    df = df.withColumn(
        "signal_line",
        exp(avg(ln("macd_line")).over(window_signal))
    ).withColumn(
        "macd_histogram", 
        col("macd_line") - col("signal_line")
    )
    
    # Cleanup intermediate columns
    if cleanup:
        df = df.drop("ema_fast", "ema_slow", "year", "month")
    
    return df

def calculate_trading_signals(
    df: DataFrame, 
    include_indicators: Optional[list] = None
) -> DataFrame:
    """
    Calculate trading signals with memory optimization.
    
    Args:
        df: Input DataFrame
        include_indicators: List of indicators to include ['rsi', 'macd']
    """
    # Default to all indicators if none specified
    if include_indicators is None:
        include_indicators = ['rsi', 'macd']
    
    # Calculate only requested indicators
    if 'rsi' in include_indicators:
        df = calculate_rsi(df, cleanup=True)
    if 'macd' in include_indicators:
        df = calculate_macd(df, cleanup=True)
    
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

def cleanup_dataframe(df: DataFrame, columns_to_keep: list) -> DataFrame:
    """
    Utility function to cleanup DataFrame by keeping only specified columns.
    
    Args:
        df: Input DataFrame
        columns_to_keep: List of column names to keep
    """
    all_columns = df.columns
    columns_to_drop = [col for col in all_columns if col not in columns_to_keep]
    return df.drop(*columns_to_drop)