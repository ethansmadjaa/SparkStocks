import streamlit as st
from exploration import explore_data
from preprocessing import preprocess_data
from analysis import analyze_data
from utils.spark_utils import create_spark_session
import yfinance as yf


def main():
    # Page config
    st.set_page_config(
        page_title="Nasdaq Tech Stocks Analysis",
        page_icon="📈",
        layout="wide"
    )

    # Initialize Spark with optimized configuration
    spark = create_spark_session(
        config={
            "spark.sql.shuffle.partitions": "10",
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.debug.maxToStringFields": "100"
        }
    )

    # Streamlit UI
    st.title("📈 Nasdaq Tech Stocks Analysis")
    st.markdown("""
    This application provides comprehensive analysis of Nasdaq technology stocks using 
    Apache Spark for processing and Streamlit for visualization.
    """)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # User input for ticker
        ticker = st.text_input("Enter stock ticker:", "AAPL").upper()

        # Time period selection
        st.subheader("Select Time Period")
        period_options = {
            "1 Month": 30,
            "6 Months": 180,
            "Year to Date": "ytd",
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825,
            "Max": None
        }
        selected_period = st.selectbox("Choose time period:", list(period_options.keys()))

        # Verify if ticker exists
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            st.success(f"Analyzing {info['longName']} ({ticker})")
        except:
            st.error("Invalid ticker symbol. Please enter a valid stock symbol.")
            return

        # Choose which operation to perform
        action = st.selectbox(
            "Choose Analysis Type",
            ["Exploration", "Preprocessing", "Analysis and Visualization"]
        )

    # Main content area
    try:
        # Handle Actions
        if action == "Exploration":
            explore_data(spark, ticker, period_options[selected_period])
        elif action == "Preprocessing":
            preprocess_data(spark, ticker, period_options[selected_period])
        elif action == "Analysis and Visualization":
            analyze_data(spark, ticker, period_options[selected_period])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try again with different parameters or contact support.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit, Apache Spark, and Python</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
