from pyspark.sql import SparkSession


def create_spark_session(config=None):
    """Create a Spark session with optional configuration."""
    builder = SparkSession.builder.appName("Stock Analysis")
    
    # Add custom configuration if provided
    if config:
        for key, value in config.items():
            builder = builder.config(key, value)
    
    spark = builder.getOrCreate()
    
    return spark
