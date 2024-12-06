import streamlit as st
from exploration import explore_data
from preprocessing import preprocess_data
from analysis import analyze_data
from utils.spark_utils import create_spark_session

# Initialize Spark
spark = create_spark_session()

# Streamlit UI
st.title("Nasdaq Tech Stocks Analysis")

# User input for ticker
ticker = st.text_input("Enter stock ticker:", "AAPL")

# Choose which operation to perform
action = st.selectbox("Choose action", ["Exploration", "Preprocessing", "Analysis and Visualization"])

# Handle Actions
if action == "Exploration":
    explore_data(spark, ticker)
elif action == "Preprocessing":
    preprocess_data(spark, ticker)
elif action == "Analysis and Visualization":
    analyze_data(spark, ticker)
