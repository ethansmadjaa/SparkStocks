from pyspark.sql import SparkSession


def create_spark_session(config=None):
    """Create a Spark session with optimized configuration."""
    default_config = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.shuffle.partitions": "8",
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
        "spark.driver.maxResultSize": "2g",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.driver.extraJavaOptions": "-XX:ReservedCodeCacheSize=256M"
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
