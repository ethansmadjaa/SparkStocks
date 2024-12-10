# Technical Analysis Parameters
TECHNICAL_PARAMS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2
}

# Spark Configuration
SPARK_CONFIG = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.shuffle.partitions": "8",
    "spark.driver.memory": "2g"
} 