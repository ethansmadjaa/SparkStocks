from pyspark.sql import SparkSession


def create_spark_session(config=None):
    """Create a Spark session with optimized configuration."""
    default_config = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.shuffle.partitions": "8",
        "spark.driver.memory": "2g",
        "spark.executor.memory": "2g",
        "spark.driver.maxResultSize": "1g",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.shuffle.partitions": "10",
        "spark.sql.window.exec.buffer.spill.threshold": "10000"
    }
    
    builder = SparkSession.builder.appName("Stock Analysis")
    
    # Add default configuration
    for key, value in default_config.items():
        builder = builder.config(key, value)
    
    # Add custom configuration if provided
    if config:
        for key, value in config.items():
            builder = builder.config(key, value)
    
    return builder.getOrCreate()
