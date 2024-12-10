import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql.functions import (
    col, date_trunc, max, min, expr, when, row_number,
    first, last
)
from pyspark.sql.window import Window
from datetime import datetime, timedelta
from utils.constants import STOCK_CATEGORIES
from exploration import get_stock_data
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType

def calculate_period_returns(spark, start_date, period="month", top_n=5):
    """
    Calculate returns for all stocks in a given period.
    
    Args:
        spark: SparkSession
        start_date: datetime object for start of analysis
        period: 'month' or 'year'
        top_n: number of top performers to return
    
    Returns:
        DataFrame with top performing stocks and their returns
    """
    # Calculate end date based on period
    if period == "month":
        end_date = start_date + timedelta(days=30)
        period_name = "Monthly"
    else:  # year
        end_date = start_date + timedelta(days=365)
        period_name = "Yearly"
    
    # Get all stock tickers
    all_stocks = []
    for category in STOCK_CATEGORIES.values():
        all_stocks.extend(category.keys())
    
    # Calculate returns for each stock
    returns_data = []
    
    for ticker in all_stocks:
        try:
            # Get stock data
            df = get_stock_data(spark, ticker)
            
            # Filter for the specific period
            df_period = df.filter(
                (col("Date") >= start_date) & 
                (col("Date") <= end_date)
            )
            
            # Calculate first and last prices properly
            first_last_prices = df_period.groupBy().agg(
                min("Date").alias("start_date"),
                max("Date").alias("end_date"),
                first("Close").alias("start_price"),
                last("Close").alias("end_price")
            ).first()
            
            if first_last_prices and first_last_prices["start_price"] and first_last_prices["end_price"]:
                # Calculate return
                return_rate = ((first_last_prices["end_price"] - first_last_prices["start_price"]) / 
                             first_last_prices["start_price"] * 100)
                
                returns_data.append({
                    "ticker": ticker,
                    "start_date": first_last_prices["start_date"],
                    "end_date": first_last_prices["end_date"],
                    "start_price": first_last_prices["start_price"],
                    "end_price": first_last_prices["end_price"],
                    "return_rate": return_rate
                })
        except Exception as e:
            st.warning(f"Could not process {ticker}: {str(e)}")
    
    if not returns_data:
        st.error("No data available for the selected period")
        return None, period_name
    
    # Convert to Spark DataFrame with explicit schema
    schema = StructType([
        StructField("ticker", StringType(), False),
        StructField("start_date", TimestampType(), True),
        StructField("end_date", TimestampType(), True),
        StructField("start_price", DoubleType(), True),
        StructField("end_price", DoubleType(), True),
        StructField("return_rate", DoubleType(), True)
    ])
    
    returns_df = spark.createDataFrame(returns_data, schema=schema)
    
    # Rank stocks by return rate
    window_spec = Window.orderBy(col("return_rate").desc())
    ranked_returns = returns_df.withColumn("rank", row_number().over(window_spec))
    
    # Get top performers
    top_performers = ranked_returns.filter(col("rank") <= top_n)
    
    return top_performers, period_name

def plot_top_performers(df, period_name):
    """Create visualization for top performing stocks."""
    # Convert to pandas for plotting
    pdf = df.toPandas()
    
    # Create figure
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f"Top {len(pdf)} {period_name} Returns",
                                      "Price Comparison"),
                        vertical_spacing=0.15,
                        row_heights=[0.4, 0.6])
    
    # Add returns bar chart
    fig.add_trace(
        go.Bar(x=pdf["ticker"],
               y=pdf["return_rate"],
               text=pdf["return_rate"].round(2).astype(str) + '%',
               textposition='auto',
               name="Return Rate"),
        row=1, col=1
    )
    
    # Add start and end prices
    for _, row in pdf.iterrows():
        fig.add_trace(
            go.Scatter(x=[row["start_date"], row["end_date"]],
                      y=[row["start_price"], row["end_price"]],
                      name=row["ticker"],
                      mode="lines+markers"),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Top Performing Stocks ({period_name})"
    )
    
    fig.update_yaxes(title_text="Return Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_xaxes(title_text="Stock", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig

def analyze_data(spark, ticker, days: int = 365):
    """Main analysis function."""
    st.subheader(f"📊 Analyzing data for {ticker}")
    
    # Add period performance analysis
    st.write("### Period Performance Analysis")
    
    # Date selection
    selected_date = st.date_input(
        "Select Start Date",
        value=datetime.now() - timedelta(days=30),
        max_value=datetime.now() - timedelta(days=1)
    )
    
    # Period selection
    period = st.radio(
        "Select Analysis Period",
        ["month", "year"],
        format_func=lambda x: x.capitalize()
    )
    
    # Number of top performers to show
    top_n = st.slider("Number of top performers to show", 3, 10, 5)
    
    if st.button("Analyze Period Performance"):
        with st.spinner("Calculating returns..."):
            # Get top performers
            top_performers, period_name = calculate_period_returns(
                spark, selected_date, period, top_n
            )
            
            if top_performers is not None:
                # Display results
                st.write(f"#### Top {top_n} Performing Stocks")
                st.dataframe(
                    top_performers.select(
                        "ticker", "return_rate", "start_price", "end_price"
                    ).toPandas()
                )
                
                # Create visualization
                fig = plot_top_performers(top_performers, period_name)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add insights
                best_performer = top_performers.first()
                if best_performer:
                    st.success(f"""
                    ### Key Insights
                    - Best performing stock: {best_performer['ticker']}
                    - Return rate: {best_performer['return_rate']:.2f}%
                    - Period: {period_name}
                    - Start date: {best_performer['start_date'].strftime('%Y-%m-%d')}
                    - End date: {best_performer['end_date'].strftime('%Y-%m-%d')}
                    """)
