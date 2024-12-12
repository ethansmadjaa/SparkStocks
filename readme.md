# Big Data Analysis Project

### Nasdaq Tech Stocks Analysis using Streamlit, Spark, and Python

This project aims to build a Python application that provides valuable insights into Nasdaq technology stocks using the Apache Spark framework. The application is intended to assist traders and investors in making informed decisions by exploring, preprocessing, and analyzing historical stock data. The project is built with **Streamlit** for visualization, **Apache Spark** for data processing, and **yfinance** for fetching stock market data.

## Project Overview

This project goes through the entire data analysis process, from data **exploration** to **preprocessing**, and finally to **analysis and visualization**. We leverage **Spark** to efficiently handle large datasets, while **Streamlit** allows us to create an interactive web interface for users.

The main goal of this application is to extract useful information from historical stock prices to aid in decision-making for investors looking to invest in these stocks.

## Objectives

- Apply big data concepts to create a structured data analysis pipeline.
- Use Spark to build a scalable solution for analyzing large datasets of stock data.
- Generate meaningful insights to assist traders in making informed investment decisions.
- Use visualization tools (Matplotlib, Seaborn, Plotly) to effectively present findings.

## Project Structure

The project is modular, consisting of the following main components:

- **app.py**: The main application file that serves as the entry point. It connects to **Streamlit** and **Spark** and facilitates user interaction through a web interface.
- **exploration.py**: Handles initial data exploration, including descriptive statistics, data overview, and insights about the data structure.
- **preprocessing.py**: Contains all preprocessing steps, such as handling missing values, data normalization, and ensuring data consistency.
- **analysis.py**: Focuses on performing financial calculations (e.g., daily returns, moving averages) and generating visualizations to assist in data analysis.
- **utils/spark\_utils.py**: Helper functions for initializing Spark and managing configurations.
- **requirements.txt**: Lists all the necessary dependencies required for the project.

## Main Features

- **Exploration Module**

  - Show the first and last 40 rows of the dataset.
  - Display the number of observations.
  - Calculate and present descriptive statistics (e.g., min, max, mean, standard deviation).
  - Identify the frequency of data points automatically.

- **Preprocessing Module**

  - Handle missing values by filling or dropping them as needed.
  - Normalize numerical columns for easier analysis.
  - Manage data consistency and formatting.

- **Analysis and Visualization Module**

  - Calculate the daily returns of each stock.
  - Plot distribution plots for daily returns to understand volatility.
  - Implement moving averages to identify trends.
  - Visualize correlations between different stocks.
  - Provide interactive data visualizations using **Seaborn**, **Matplotlib**, and **Plotly**.

## Installation and Usage

### Prerequisites

- Python 3.11
- Apache Spark

### Installation Steps

1. Clone the repository:

   ```sh
   git clone https://github.com/ethansmadjaa/SparkStocks.git
   cd SparkStocks
   ```

2. Set up a virtual environment and activate it:

   On MacOS:

   ```sh
   python3.11 -m venv .venv
   source .venv/bin/activate  
   ```
   on Windows
   ```sh
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
   

4. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

5. Run the Streamlit app:

   ```sh
   streamlit run app.py
   ```

### Using the Application

- **Ticker Input**: Enter a stock ticker (e.g., "AAPL" for Apple Inc.) to begin analysis.
- **Exploration**: Explore initial data insights, including descriptive statistics and general information.
- **Preprocessing**: Perform preprocessing steps to clean the data and handle any inconsistencies.
- **Analysis**: View detailed analysis, including moving averages, daily returns, and visual insights to assist in understanding trends.

## Example Use Case

A trader wishes to analyze the performance of Apple Inc. over the past year. They enter the ticker symbol "AAPL" and select different modules for exploration, preprocessing, and analysis. The application provides descriptive statistics, identifies trends using moving averages, and visualizes the daily returns distribution, aiding the trader in making informed investment decisions.


## License

This project is licensed under the MIT License.

## Acknowledgements

- **Apache Spark** for scalable data processing.
- **Streamlit** for providing a user-friendly web application framework.
- **Yahoo Finance (yfinance)** for the stock market data.

---

Feel free to reach out if you have any questions or run into issues using the app. Happy analyzing!
