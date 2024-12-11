import yfinance as yf
from datetime import datetime, timedelta, date
from llm import retrieval_step, generation_step
import re
from funcs import write_debug_log
import math
from dateutil.relativedelta import relativedelta
import pandas as pd

# I want to use stock performance to generate baseline performance labels for periods of time, 
# and then ask the LLM "how did {insert company} perform during the period {insert time period}" 
# and compare the LLMs answer to the answer gotten using stock price analysis. This could be automated 
# by simple taking a sliding window and asking about every period in the sliding window, and simultaneously 
# querying finance for that data and making assumptions based on conditions.

# Global debug log name to write debug statements
debug_file = "test_new_auto_gen_eval_log.md"

# Function to evaluate performance based on calculated values
def evaluate_performance(eps_growth, revenue_growth, income_growth):
    # Determine performance
    if ((eps_growth > 0 and revenue_growth > 0) or 
        (eps_growth > 0 and income_growth > 0) or 
        (revenue_growth > 0 and income_growth > 0)):
        return True  # The company performed well
    else:
        return False  # The company did not perform well

def get_income_statement(ticker):
    ticker_object = yf.Ticker(ticker)
    income_stmt = ticker_object.quarterly_income_stmt
    return income_stmt

def is_in_daterange(current_date, start_date, end_date):
    """
    Check if the current_date is within the range start_date to end_date (inclusive).

    Args:
        current_date (datetime): The date to check.
        start_date (datetime): The start of the date range.
        end_date (datetime): The end of the date range.

    Returns:
        bool: True if current_date is within the range (inclusive), False otherwise.
    """
    return start_date <= current_date <= end_date


# Function to grab stock prices from yfinance
def get_stock_prices(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    ticker_hist = stock.history(interval='1d', start=start_date, end=end_date)
    return ticker_hist

# Function to analyze company performance over time period
def get_company_performance(ticker, start_date, end_date, verbose = False):
    income_stmt = get_income_statement(ticker)
    if income_stmt.empty:
        write_debug_log("No stock data found for the given ticker and date range.", log_file=debug_file, with_timestamp=False, print_message=True)
        
        return

    # Calculate performance metrics
    metrics = ['Diluted EPS', 'Total Revenue', 'Net Income']
    growth_data = {metric: 0 for metric in metrics}
    for current_quarter in income_stmt.columns:
        if(not is_in_daterange(current_quarter, start_date, end_date)):
            continue
        else:
            if verbose: write_debug_log(f"HERE: current_quarter {current_quarter} is in the range {start_date} to {end_date}", log_file=debug_file, with_timestamp=False, print_message=True)
        
        current_value = income_stmt[current_quarter]
        try:
            # Get the corresponding quarter from the previous year
            prior_quarter = current_quarter - pd.DateOffset(years=1)
            if verbose: print(f"prior: {prior_quarter}")

            if prior_quarter in income_stmt.columns:
                # Calculate growth for each metric
                return_growth = True
                for metric in metrics:
                    current_value = income_stmt[current_quarter].loc[metric]
                    prior_value = income_stmt[prior_quarter].loc[metric]
                    
                    if math.isnan(prior_value):
                        return_growth = False
                        continue
                        
                    if verbose: write_debug_log(f"curval for {metric}: {current_value}", log_file=debug_file, with_timestamp=False, print_message=True)
                    if verbose: write_debug_log(f"priorval for {metric}: {prior_value}", log_file=debug_file, with_timestamp=False, print_message=True)

                    if prior_value != 0:  # Avoid division by zero
                        growth = ((current_value - prior_value) / prior_value) * 100
                    else:
                        growth = None  # Handle zero prior value

                    # Append the result
                    growth_data[metric] = growth
                    
                    
                print(growth_data)
                if return_growth: return evaluate_performance(growth_data[metrics[0]], growth_data[metrics[1]], growth_data[metrics[2]])
                else: return None
            else:
                if verbose: print(f"{current_quarter} does not have prior data")
        except Exception as e:
            print(f"Error processing quarter {current_quarter}: {e}")
    return None
    


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

def generation_eval(companies, years, k, verbose = False):
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
                    end_date = start_date + timedelta(days=92)
                    if company['offset_year']:
                        end_date = end_date + relativedelta(years=-1)
                        start_date = start_date + relativedelta(years=-1)
                    # today = date.today()
                    # if end_date.date() > today:
                    #     write_debug_log(f"end date {end_date}after today", log_file=debug_file, with_timestamp=False, print_message=True)
                    #     continue
                    
                    if verbose: write_debug_log(f"Automated evaluation for {company['name']} for Q{quarter} of {year} ({start_date.strftime('%m/%d/%Y')} - {end_date.strftime('%m/%d/%Y')}):\n", log_file=debug_file, with_timestamp=False, print_message=True)

                    
                    performance = get_company_performance(company['ticker'], start_date, end_date, verbose = verbose)
                    
                    if performance == None:
                        write_debug_log(f"Not generating for Q{quarter} of {year} during {start_date} to {end_date}\n\n", log_file=debug_file, with_timestamp=False, print_message=True)
                        continue
                    
                    query = f"Did {company['name']} perform well in Q{quarter} of {year} based on their Earnings Per Share EPS, Total Revenue, and Net Income?"
                    top_n = retrieval_step(message = query, hybrid_search=True, n = k)
                    
                    for j, context in enumerate(top_n):
                        write_debug_log(f"Context {j}: {context}\n\n", log_file=debug_file, with_timestamp=False, print_message=True)

                    
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
                            write_debug_log(f'LLM response: {answer}\n', log_file=debug_file, with_timestamp=False, print_message=True)
                    else:
                        write_debug_log(f'ERROR WITH REGULAR EXPRESSION MATCHING ON RETURNED ANSWER: {answer}', log_file=debug_file, with_timestamp=False, print_message=True)
                        failure = True

                    
                    company_exists = company_counts.get(company['name'], False)
                    if not company_exists and not failure:
                        company_counts[company['name']] = {
                            'count_correct': 0,
                            'count_total': 0
                        }
                    if not failure and performance == value:
                        count_correct += 1
                        company_counts[company['name']]['count_correct'] += 1
                        write_debug_log("LLM performance evaluation Was a SUCCESS\n", log_file=debug_file, with_timestamp=False, print_message=True)
                    else:
                        write_debug_log("LLM performance evaluation Was a FAILURE\n", log_file=debug_file, with_timestamp=False, print_message=True)
                        
                    if not failure:
                        count_total += 1
                        company_counts[company['name']]['count_total'] += 1
                        write_debug_log(message="", log_file=debug_file, with_timestamp=False, print_message=True)
                        write_debug_log("-"*128, log_file=debug_file, with_timestamp=False, print_message=True)
                        write_debug_log(message="",  log_file=debug_file, with_timestamp=False, print_message=True)
        write_debug_log(f"total correct = {count_correct}", log_file=debug_file, with_timestamp=False, print_message=True)
        write_debug_log(f"total correct percentage = {count_correct/count_total}", log_file=debug_file, with_timestamp=False, print_message=True)
        write_debug_log(message="", log_file=debug_file, with_timestamp=False, print_message=True)
        for c_name, c_count in company_counts.items():
            
            write_debug_log(f"{c_name}: {c_count['count_correct']}/{c_count['count_total']} - {c_count['count_correct']/c_count['count_total']}", log_file=debug_file, with_timestamp=False, print_message=True)
        write_debug_log("\n"*6, log_file=debug_file, with_timestamp=False, print_message=True)
        
        return count_correct, count_total, company_counts
                    
                

                

                
                
                
    except Exception as e:
        write_debug_log(f"An error occurred: {e}", log_file=debug_file, with_timestamp=False, print_message=True)

if __name__ == "__main__":
    # ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL, TSLA): ")
    companies = [
                {
                    'name':'Tesla',
                    'ticker':'tsla',
                    "fy_start":datetime(year = 2024, day=1, month=1),
                    "offset_year":False
                },
                {
                   'name':'Apple',
                   'ticker':'aapl',
                   "fy_start":datetime(year = 2024, day=1, month=10),
                    "offset_year":True
                },
                {
                   'name':'Nvidia',
                   'ticker':'nvda',
                   "fy_start":datetime(year = 2024, day=1, month=2),
                    "offset_year":False
                },
                {
                   'name':'Microsoft',
                   'ticker':'msft',
                   "fy_start":datetime(year = 2024, day=1, month=7),
                    "offset_year":True
                },
                {
                   'name':'Meta',
                   'ticker':'meta',
                   "fy_start":datetime(year = 2024, day=1, month=1),
                   "blacklisted_quarters": [
                       ("Q1", 2022)
                   ],
                    "offset_year":False
                },
                {
                   'name':'Amazon',
                   'ticker':'amzn',
                   "fy_start":datetime(year = 2024, day=1, month=1),
                    "offset_year":False
                },
                {
                   'name':'Berkshire Hathaway',
                   'ticker': "BRK-B",
                   "fy_start":datetime(year = 2024, day=1, month=1),
                    "offset_year":False
                },
                {
                   'name':'Google',
                   'ticker':'GOOG',
                   "fy_start":datetime(year = 2024, day=1, month=1),
                    "offset_year":False
                }
            ]
    years = [2022, 2023, 2024, 2025]
    num_trials = 10
    
    # generation_eval(companies, years, True)
    

    for k in range(1,2):
        debug_file = debug_file+f"_k_equals_{k}.md"
        
        overall_average_sum = 0
        
        overall_company_sum = {}
        
        for c in companies:
            overall_company_sum[c['name']] = {
                                'count_correct': 0,
                                'count_total': 0
                            }
        for _ in range(num_trials):
            
            correct_responses, total_count, company_counts = generation_eval(companies, years, k, True)
            overall_average_sum += correct_responses/total_count
            for c_name, c_count in company_counts.items():
                overall_company_sum[c_name]['count_correct'] += c_count['count_correct']
                overall_company_sum[c_name]['count_total'] += c_count['count_total']
            
        
    
        write_debug_log("\n"*5, log_file=debug_file, with_timestamp=False, print_message=True)
        write_debug_log(f"Overall Average Across {num_trials} trials: {overall_average_sum/num_trials}\n", log_file=debug_file, with_timestamp=False, print_message=True)
        
        for c_name, c_count in overall_company_sum.items():
            if (c_count['count_total'] > 0):
                write_debug_log(f"{c_name}: {c_count['count_correct']}/{c_count['count_total']} - {c_count['count_correct']/c_count['count_total']}", log_file=debug_file, with_timestamp=False, print_message=True)
        
    
