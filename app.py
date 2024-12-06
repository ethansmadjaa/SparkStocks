import streamlit as st
from pyspark.sql import SparkSession
import yfinance as yf


@st.cache_resource
def create_spark_session():
    sparksession = SparkSession.builder \
        .appName("Stock Analysis") \
        .getOrCreate()
    return sparksession


spark = create_spark_session()


def get_stock_data(ticker, period='1y', interval='1d'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    data.reset_index(inplace=True)
    return data


def main():
    st.title("Nasdaq Tech Stocks Analysis")

    # User input for ticker symbol
    ticker = st.text_input("Enter the stock ticker symbol:", 'AAPL')

    if ticker:
        # Fetch data using yfinance
        data = get_stock_data(ticker)

        # Convert to Spark DataFrame
        df_spark = spark.createDataFrame(data)

        # Display data in Streamlit
        st.subheader(f"Stock Data for {ticker}")
        st.write(data)

        # Show descriptive statistics using Spark
        st.subheader("Descriptive Statistics")
        describe_df = df_spark.describe().toPandas()
        st.write(describe_df)

        # Additional functionalities can be added here


if __name__ == "__main__":
    main()
