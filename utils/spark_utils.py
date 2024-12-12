from pyspark.sql import SparkSession
from typing import Optional, Dict


def create_spark_session(config: Optional[Dict[str, str]] = None) -> SparkSession:
    """
    Create a Spark session with memory-optimized configuration.
    
    Args:
        config: Optional custom configuration
    """
    default_config = {
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
        "spark.driver.maxResultSize": "2g",
        "spark.driver.extraJavaOptions": "-XX:ReservedCodeCacheSize=1024m -XX:InitialCodeCacheSize=512m",
        "spark.executor.extraJavaOptions": "-XX:ReservedCodeCacheSize=1024m -XX:InitialCodeCacheSize=512m",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.shuffle.partitions": "200",
        "spark.memory.fraction": "0.8",
        "spark.memory.storageFraction": "0.3",
        "spark.sql.window.exec.buffer.in.memory.threshold": "10000",
        "spark.sql.window.exec.buffer.spill.threshold": "10000",
        "spark.sql.autoBroadcastJoinThreshold": "10m",
        "spark.cleaner.periodicGC.interval": "1min"
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


def cleanup_spark_cache(spark: SparkSession) -> None:
    """Utility function to clean up Spark cache."""
    spark.catalog.clearCache()
