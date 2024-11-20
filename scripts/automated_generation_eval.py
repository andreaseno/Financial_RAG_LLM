import yfinance as yf
from datetime import datetime, timedelta, date
from llm import retrieval_step, generation_step
import re
from funcs import write_debug_log

# I want to use stock performance to generate baseline performance labels for periods of time, 
# and then ask the LLM "how did {insert company} perform during the period {insert time period}" 
# and compare the LLMs answer to the answer gotten using stock price analysis. This could be automated 
# by simple taking a sliding window and asking about every period in the sliding window, and simultaneously 
# querying finance for that data and making assumptions based on conditions.

# Function to evaluate performance based on calculated values
def evaluate_performance(percentage_change, volatility, average_daily_volume):
    percentage_change_threshold = 5  # in percentage
    volatility_threshold = 5  # in percentage
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
        write_debug_log("No stock data found for the given ticker and date range.", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        
        return
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
        write_debug_log(f"Performance Analysis for {ticker}:", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        write_debug_log(f"Initial Closing Price: ${initial_price:.2f}", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        write_debug_log(f"Final Closing Price: ${final_price:.2f}", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        write_debug_log(f"Percentage Change: {percentage_change:.2f}%", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        write_debug_log(f"Volatility: {volatility:.2f}%", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        write_debug_log(f"Average Daily Volume: {average_volume:.0f}\n", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
    
    return evaluate_performance(percentage_change, volatility, average_volume)


def get_quarter_start_dates(fiscal_start, year):
    """
    Given a fiscal year start date, a year, and a quarter, return
    the start datetime objects for all four quarters in the fiscal year.

    Args:
        fiscal_start (datetime): Start of the fiscal year (month and day are used).
        year (int): The specific year to calculate the quarter dates for.
        quarter (int): The quarter (1 to 4) to calculate the start date for.

    Returns:
        dict: A dictionary mapping quarters (1-4) to their start dates as datetime objects.
    """
    # Adjust the fiscal start to the given year
    fiscal_start_date = datetime(year=year, month=fiscal_start.month, day=fiscal_start.day)

    # Calculate the start dates for each quarter
    quarter_starts = {
        1: fiscal_start_date,
        2: fiscal_start_date + timedelta(days=3*30),  # Approximate 3 months
        3: fiscal_start_date + timedelta(days=6*30),  # Approximate 6 months
        4: fiscal_start_date + timedelta(days=9*30)   # Approximate 9 months
    }
    
    # Ensure we handle leap year correctly by using month delta approach
    quarter_starts = {
        q: (fiscal_start_date.replace(month=((fiscal_start.month - 1 + (q - 1) * 3) % 12 + 1),
                                      year=year + ((fiscal_start.month - 1 + (q - 1) * 3) // 12)))
        for q in range(1, 5)
    }

    return quarter_starts

def generation_eval(companies, years, verbose = False):
    try:
        count_correct = 0
        count_total = 0
        
        company_counts = {}
        
        for company in companies:
            for year in years:
                for quarter in range(1,5):
                    if company.get("blacklisted_quarters") and (f'Q{quarter}', year) in company["blacklisted_quarters"]:
                        continue
                    dates = get_quarter_start_dates(company['fy_start'], year)
                    start_date = dates[quarter]
                    end_date = start_date + timedelta(days=90)
                    
                    today = date.today()
                    if end_date.date() > today:
                        continue
                    
                    if verbose: write_debug_log(f"Automated evaluation for {company['name']} for Q{quarter} of {year} ({start_date.strftime('%m/%d/%Y')} - {end_date.strftime('%m/%d/%Y')}):\n", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)

                    
                    performance = get_company_performance(company['ticker'], start_date, end_date, verbose = verbose)
                    
                    query = f"Did {company['name']} perform well in Q{quarter} of {year}?"
                    top_n = retrieval_step(message = query, n = 5)
                    
                    query += """
                    
                    Please provide your yes/no answer as a boolean value in the same format as the <example> below and do not include any other information except what is seen in the <example>:
                    
                    <example>
                    **answer** = true
                    </example>
                    """
                    
                    answer = generation_step(message = query, top_n = top_n, eval = True)
                    
                    matches = list(re.finditer(r"\*\*answer\*\*\s*[:=]?\s*(true|false)", answer, re.IGNORECASE))

                    failure = False
                    if matches:
                        last_match = matches[-1]  # Get the last match
                        value = last_match.group(1).lower() == 'true'  # This will give you 'true' or 'false'
                        if verbose:
                            write_debug_log(f'LLM response: {answer}\n', log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
                    else:
                        write_debug_log(f'ERROR WITH REGULAR EXPRESSION MATCHING ON RETURNED ANSWER: {answer}', log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
                        failure = True

                    
                    company_exists = company_counts.get(company['name'], False)
                    if not company_exists:
                        company_counts[company['name']] = {
                            'count_correct': 0,
                            'count_total': 0
                        }
                    if performance == value and not failure:
                        count_correct += 1
                        company_counts[company['name']]['count_correct'] += 1
                        write_debug_log("LLM performance evaluation Was a SUCCESS\n", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
                    else:
                        write_debug_log("LLM performance evaluation Was a FAILURE\n", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
                        
                    if not failure:
                        count_total += 1
                        company_counts[company['name']]['count_total'] += 1
                        write_debug_log(message="", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
                        write_debug_log("-"*128, log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
                        write_debug_log(message="",  log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        write_debug_log(f"total correct = {count_correct}", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        write_debug_log(f"total correct percentage = {count_correct/count_total}", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        write_debug_log(message="", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        for c_name, c_count in company_counts.items():
            write_debug_log(f"{c_name}: {c_count['count_correct']}/{c_count['count_total']} - {c_count['count_correct']/c_count['count_total']}", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        write_debug_log("\n"*6, log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        
        return count_correct, count_total, company_counts
                    
                

                

                
                
                
    except Exception as e:
        write_debug_log(f"An error occurred: {e}", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)

if __name__ == "__main__":
    # ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL, TSLA): ")
    companies = [
                {
                    'name':'Tesla',
                    'ticker':'tsla',
                    "fy_start":datetime(year = 2024, day=1, month=1)
                    
                },
                {
                   'name':'Apple',
                   'ticker':'aapl',
                   "fy_start":datetime(year = 2024, day=1, month=10)
                },
                {
                   'name':'Nvidia',
                   'ticker':'nvda',
                   "fy_start":datetime(year = 2024, day=1, month=2),
                   "blacklisted_quarters": [
                       ("Q3", 2024)
                   ]
                },
                {
                   'name':'Microsoft',
                   'ticker':'msft',
                   "fy_start":datetime(year = 2024, day=1, month=7)
                },
                {
                   'name':'Meta',
                   'ticker':'meta',
                   "fy_start":datetime(year = 2024, day=1, month=1),
                   "blacklisted_quarters": [
                       ("Q1", 2022)
                   ]
                },
                {
                   'name':'Amazon',
                   'ticker':'amzn',
                   "fy_start":datetime(year = 2024, day=1, month=1)
                }
            ]
    years = [2022, 2023, 2024]
    
    # generation_eval(companies, years, True)
    
    num_trials = 5
    overall_average_sum = 0
    
    overall_company_sum = {}
    
    for c in companies:
        overall_company_sum[c['name']] = {
                            'count_correct': 0,
                            'count_total': 0
                        }
    
    for _ in range(num_trials):
        
        correct_responses, total_count, company_counts = generation_eval(companies, years, True)
        overall_average_sum += correct_responses/total_count
        for c_name, c_count in company_counts.items():
            overall_company_sum[c_name]['count_correct'] += c_count['count_correct']
            overall_company_sum[c_name]['count_total'] += c_count['count_total']
            
        
    
    write_debug_log("\n"*5, log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
    write_debug_log(f"Overall Average Across {num_trials} trials: {overall_average_sum/num_trials}\n", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
    
    for c_name, c_count in overall_company_sum.items():
        write_debug_log(f"{c_name}: {c_count['count_correct']}/{c_count['count_total']} - {c_count['count_correct']/c_count['count_total']}", log_file="auto_gen_eval_log.md", with_timestamp=False, print_message=True)
        
    
