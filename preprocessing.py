from pyspark.sql import DataFrame, SparkSession
import streamlit as st
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *

def preprocess_data(spark, ticker: str, days: int = 365):
    """Main function for preprocessing."""
    st.subheader(f"📊 Preprocessing data for {ticker}")
    st.info("Preprocessing module is under development. Coming soon!")
    pass
