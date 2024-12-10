# This is our main app file - it handles the UI and brings all the analysis pieces together
from datetime import datetime

import streamlit as st
import yfinance as yf

from analysis import analyze_data
from exploration import explore_data
from preprocessing import preprocess_data
from utils.constants import STOCK_CATEGORIES
from utils.spark_utils import create_spark_session


def get_ytd_days():
    # We need to know how many days have passed this year
    # This is useful for the YTD (Year-To-Date) option in our date picker
    today = datetime.now()
    start_of_year = datetime(today.year, 1, 1)  # January 1st of current year
    return (today - start_of_year).days


def get_stock_info(ticker: str) -> dict:
    # This function gets all the important details about a stock
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get the current price and previous close to calculate price change
        current_price = info.get("currentPrice", 0)
        previous_close = info.get("previousClose", current_price)
        price_change = ((current_price - previous_close) / previous_close * 100) if previous_close else 0
        
        return {
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "current_price": current_price,
            "price_change": price_change,
            "volume": info.get("volume", 0)
        }
    except:
        return None


def format_market_cap(market_cap: int) -> str:
    # This makes big numbers easier to read
    # Instead of seeing 1000000000, you'll see 1B
    # We handle four cases: Trillions, Billions, Millions, and anything smaller
    if market_cap >= 1e12:
        return f"${market_cap / 1e12:.1f}T"  # Trillions
    elif market_cap >= 1e9:
        return f"${market_cap / 1e9:.1f}B"   # Billions
    elif market_cap >= 1e6:
        return f"${market_cap / 1e6:.1f}M"   # Millions
    else:
        return f"${market_cap:,.0f}"         # Regular numbers with commas


def main():
    # This is where everything comes together
    # We set up the page, create the UI, and handle all user interactions

    # First, let's set up how our app looks
    st.set_page_config(
        page_title="Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide",  # Use the full screen width
        initial_sidebar_state="expanded"  # Start with the sidebar open
    )

    # Add some custom CSS to make everything look better
    # This makes the spacing and padding just right
    st.markdown("""
        <style>
        .main { padding: 0rem 1rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
        
        /* Improved styling for stock details in sidebar */
        .stock-info {
            background-color: #262730;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #464B5C;
        }
        .stock-info h4 {
            color: #FFFFFF;
            margin: 0 0 10px 0;
            font-size: 1.1em;
        }
        .stock-info p {
            color: #FAFAFA;
            margin: 5px 0;
            font-size: 0.9em;
        }
        .stock-info .highlight {
            color: #00CC96;
            font-weight: bold;
        }
        .stock-info .label {
            color: #9BA1B9;
            margin-right: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main title of our app
    st.title("📈 Stock Analysis Dashboard")

    # We use session_state to remember which stocks the user is interested in
    # This persists even when the app refreshes
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []

    # Now we set up the sidebar - this is where users pick their stocks
    st.sidebar.title("Stock Selection")
    
    # Users can pick stocks in three ways:
    # 1. From categories (like Tech, Healthcare)
    # 2. By typing in a stock symbol
    # 3. From their saved portfolio
    selection_method = st.sidebar.radio(
        "How do you want to pick a stock?",
        ["Category", "Custom Ticker", "Portfolio"],
        key="main_selection_method"
    )

    # Based on how they want to pick their stock, we show different options
    if selection_method == "Category":
        # If they want to browse categories, we show them organized groups of stocks
        category = st.sidebar.selectbox(
            "Pick a category",
            list(STOCK_CATEGORIES.keys()),
            key="main_category"
        )
        stock_options = STOCK_CATEGORIES[category]
        selected_stock = st.sidebar.selectbox(
            "Choose your stock",
            list(stock_options.keys()),
            format_func=lambda x: f"{x} - {stock_options[x]}",  # Show both symbol and name
            key="main_stock"
        )
    elif selection_method == "Portfolio":
        # If they want to use their portfolio, but haven't added any stocks yet
        if not st.session_state.portfolio:
            st.sidebar.warning("Add some stocks to your portfolio first!")
            selected_stock = "AAPL"  # Default to Apple if portfolio is empty
        else:
            # Let them pick from their saved stocks
            selected_stock = st.sidebar.selectbox(
                "Pick from your portfolio",
                st.session_state.portfolio,
                key="portfolio_stock"
            )
    else:
        # If they want to type in their own stock symbol
        selected_stock = st.sidebar.text_input(
            "Type in a stock symbol",
            value="AAPL",
            max_chars=5,  # Stock symbols are usually 1-5 characters
            key="main_ticker"
        ).upper()  # Convert to uppercase since that's how stock symbols are written

    # Portfolio management section
    st.sidebar.markdown("---")  # Add a visual separator
    st.sidebar.subheader("Your Portfolio")

    # Let users add the current stock to their portfolio
    if st.sidebar.button("Add Current Stock"):
        if selected_stock not in st.session_state.portfolio:
            st.session_state.portfolio.append(selected_stock)
            st.sidebar.success(f"Added {selected_stock}!")

    # Show all the stocks in their portfolio with remove buttons
    if st.session_state.portfolio:
        st.sidebar.write("Stocks you're tracking:")
        for stock in st.session_state.portfolio:
            col1, col2 = st.sidebar.columns([3, 1])  # 3:1 ratio looks better
            col1.write(f"• {stock}")
            if col2.button("🗑️", key=f"remove_{stock}"):  # Trash can emoji for remove
                st.session_state.portfolio.remove(stock)
                st.sidebar.success(f"Removed {stock}")

    # Time range selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Time Range")

    # Define our preset time periods
    period_options = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "YTD": get_ytd_days(),  # Days since January 1st
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825
    }

    # Let users choose between preset periods or a custom number of days
    period_mode = st.sidebar.radio(
        "How do you want to set the time range?",
        ["Preset", "Custom"],
        key="period_mode"
    )

    if period_mode == "Preset":
        # Use a slider for preset periods
        selected_period = st.sidebar.select_slider(
            "Pick a time range",
            options=list(period_options.keys())
        )
        days = period_options[selected_period]
    else:
        # Let them input any number of days
        days = st.sidebar.number_input(
            "How many days of data?",
            min_value=1,
            max_value=3650,  # About 10 years
            value=365  # Default to 1 year
        )

    # Show detailed information about the selected stock
    stock_info = get_stock_info(selected_stock)
    if stock_info:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Stock Details")
        # Display the info in a nice formatted box
        st.sidebar.markdown(f"""
        <div class="stock-info">
        <h4>{selected_stock} - {stock_info['name']}</h4>
        <p><span class="label">Price:</span> <span class="highlight">${stock_info['current_price']:.2f}</span></p>
        <p><span class="label">Change:</span> <span class="highlight">{stock_info['price_change']:.2f}%</span></p>
        <p><span class="label">Volume:</span> {stock_info['volume']:,}</p>
        <p><span class="label">Market Cap:</span> ${format_market_cap(stock_info['market_cap'])}</p>
        <p><span class="label">Sector:</span> {stock_info['sector']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Start up our Spark session - this is what powers our big data processing
    spark = create_spark_session()

    # Create three main tabs for different types of analysis
    tab1, tab2, tab3 = st.tabs(["Explore", "Process", "Analyze"])

    # Run the appropriate analysis based on which tab is selected
    with tab1:
        explore_data(spark, selected_stock, days)
    with tab2:
        preprocess_data(spark, selected_stock, days)
    with tab3:
        analyze_data(spark, selected_stock, days)

    # Always clean up our Spark session when we're done
    spark.stop()


# This is where the app starts running
if __name__ == "__main__":
    main()
