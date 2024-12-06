from pyspark.sql import SparkSession


def create_spark_session():
    spark = SparkSession.builder.appName("Stock Analysis").getOrCreate()
    return spark
