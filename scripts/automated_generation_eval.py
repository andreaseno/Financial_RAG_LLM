import yfinance as yf
from datetime import datetime

# I want to use stock performance to generate baseline performance labels for periods of time, 
# and then ask the LLM "how did {insert company} perform during the period {insert time period}" 
# and compare the LLMs answer to the answer gotten using stock price analysis. This could be automated 
# by simple taking a sliding window and asking about every period in the sliding window, and simultaneously 
# querying finance for that data and making assumptions based on conditions.

# Function to evaluate performance based on calculated values
def evaluate_performance(percentage_change, volatility, average_daily_volume):
    percentage_change_threshold = 10  # in percentage
    volatility_threshold = 20  # in percentage
    average_daily_volume_threshold = 1000000  # in Shares

    # Determine performance
    if (percentage_change > percentage_change_threshold and
        volatility < volatility_threshold and
        average_daily_volume > average_daily_volume_threshold):
        return True  # The company performed well
    else:
        return False  # The company did not perform well

# Function to grab stock prices from yfinance
def get_stock_prices(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    ticker_hist = stock.history(interval='1d', start=start_date, end=end_date)
    return ticker_hist

# Function to analyze company performance over time period
def get_company_performance(ticker, start_date, end_date, verbose = False):
    stock_prices = get_stock_prices(ticker, start_date, end_date)
    if stock_prices.empty:
        print("No stock data found for the given ticker and date range.")
        return
    # print(stock_prices)
    # Calculate performance metrics
    initial_price = stock_prices['Close'].iloc[0]
    final_price = stock_prices['Close'].iloc[-1]
    percentage_change = ((final_price - initial_price) / initial_price) * 100

    # Calculate volatility (standard deviation of daily returns)
    daily_returns = stock_prices['Close'].pct_change()
    volatility = daily_returns.std() * 100  # in percentage

    # Average trading volume
    average_volume = stock_prices['Volume'].mean()

    # Output performance analysis
    # This is a good place to start, could expand on this by doing comparative analysis to other stocks in the same industry
    if verbose:
        print(f"Performance Analysis for {ticker}:")
        print(f"Initial Closing Price: ${initial_price:.2f}")
        print(f"Final Closing Price: ${final_price:.2f}")
        print(f"Percentage Change: {percentage_change:.2f}%")
        print(f"Volatility: {volatility:.2f}%")
        print(f"Average Daily Volume: {average_volume:.0f}")\
    
    return evaluate_performance(percentage_change, volatility, average_volume)

if __name__ == "__main__":
    # ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL, TSLA): ")
    tickers = ['tsla','aapl','nvda','msft','meta','NVCR','AMC','HE']
    years = [2023, 2024]
    try:
        for ticker in tickers:
            for year in years:
                end_date = datetime(year, 12, 31, 16, 00)
                start_date = datetime(year, 1, 1, 9, 30)
                performance = get_company_performance(ticker, start_date, end_date, verbose=True)
                # print(f"the returned value was {performance}\n")
                
    except Exception as e:
        print(f"An error occurred: {e}")
