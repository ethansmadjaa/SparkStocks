import streamlit as st
from exploration import explore_data
from preprocessing import preprocess_data
from analysis import analyze_data
from utils.spark_utils import create_spark_session
from utils.constants import STOCK_CATEGORIES
import yfinance as yf

def get_stock_info(ticker: str) -> dict:
    """Get basic stock information."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "current_price": info.get("currentPrice", 0)
        }
    except:
        return None


def format_market_cap(market_cap: int) -> str:
    """Format market cap in billions/millions."""
    if market_cap >= 1e12:
        return f"${market_cap / 1e12:.1f}T"
    elif market_cap >= 1e9:
        return f"${market_cap / 1e9:.1f}B"
    elif market_cap >= 1e6:
        return f"${market_cap / 1e6:.1f}M"
    else:
        return f"${market_cap:,.0f}"


def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

    st.title("📈 Stock Analysis Dashboard")

    # Sidebar for stock selection
    st.sidebar.title("Stock Selection")

    # Method selection
    selection_method = st.sidebar.radio(
        "Select stock by:",
        ["Category", "Custom Ticker"],
        key="main_selection_method"
    )

    if selection_method == "Category":
        # Category selection
        category = st.sidebar.selectbox(
            "Select Category",
            list(STOCK_CATEGORIES.keys()),
            key="main_category"
        )

        # Create a formatted selection for stocks in category
        stock_options = STOCK_CATEGORIES[category]
        selected_stock = st.sidebar.selectbox(
            "Select Stock",
            list(stock_options.keys()),
            format_func=lambda x: f"{x} - {stock_options[x]}",
            key="main_stock"
        )
    else:
        # Custom ticker input with validation
        selected_stock = st.sidebar.text_input(
            "Enter Stock Ticker",
            value="AAPL",
            max_chars=5,
            key="main_ticker"
        ).upper()

    # Get stock info
    stock_info = get_stock_info(selected_stock)

    if stock_info:
        # Display stock information
        st.sidebar.markdown("---")
        st.sidebar.subheader("Stock Information")
        st.sidebar.markdown(f"""
        **{stock_info['name']}** ({selected_stock})
        - Sector: {stock_info['sector']}
        - Industry: {stock_info['industry']}
        - Market Cap: {format_market_cap(stock_info['market_cap'])}
        - Current Price: ${stock_info['current_price']:.2f}
        """)

    # Time period selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Time Period")
    period_options = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825
    }
    selected_period = st.sidebar.select_slider(
        "Select Analysis Period",
        options=list(period_options.keys())
    )
    days = period_options[selected_period]

    # Create Spark session
    spark = create_spark_session()

    # Main content
    tab1, tab2, tab3 = st.tabs(["Exploration", "Preprocessing", "Analysis"])

    with tab1:
        explore_data(spark, selected_stock, days)

    with tab2:
        preprocess_data(spark, selected_stock, days)

    with tab3:
        analyze_data(spark, selected_stock, days)

    # Clean up Spark session
    spark.stop()


if __name__ == "__main__":
    main()
