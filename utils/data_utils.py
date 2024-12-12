from datetime import datetime, timedelta
import yfinance as yf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import *


def get_stock_data(spark: SparkSession, ticker: str, days: int = 365) -> DataFrame:
    """Fetch stock data and convert to Spark DataFrame with optimized configuration."""
    # Configure Spark for better performance
    spark.conf.set("spark.sql.shuffle.partitions", "8")  # Adjust based on your data size
    spark.conf.set('spark.sql.execution.arrow.pyspark.enabled', "true")

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
