import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import ollama
import json
import psycopg2
import ast
import re
from pgvector_db_funcs import retrieve_n

# Connection parameters for pgvector
host = "localhost"         # Hostname (since you've mapped the container port)
port = "5432"              # Port you mapped (5432)
dbname = "vectordb"     # Name of your database
user = "admin"           # Username you provided during setup
password = "adminpass"   # Password you provided during setup

# Initialize the Ollama client
client = ollama.Client()

retrieved_files_path = "retrieved_documents.md"

system_prompt = """You are an AI assistant tasked with answering financial questions. Your task is to answer simple questions about a company based on the <context> element of the query.
    Here is an example query in the same format queries will be asked:
    
    ```
    **Context**: <context>
                ...
                <context>
    **Query**: <query>
                ...
                </query>
    ```
    
    When the user asks a question about a company, use information from the <context> element to answer the question asked in <query>. Assume that the <context> information is relevant to the company and time period asked about, even if not explicitly stated.
    
    
    
    
"""

def extract_query_details(query):
    # List of valid companies and their stock tickers
    companies = [
        'Tesla', 'Apple', 'Nvidia', 'Microsoft', 'Meta', 'Amazon', 'Google', 'Berkshire Hathaway'
    ]
    stock_tickers = {
        'Tesla': 'TSLA',
        'Apple': 'AAPL',
        'Nvidia': 'NVDA',
        'Microsoft': 'MSFT',
        'Meta': 'META',
        'Amazon': 'AMZN',
        'Google': 'GOOGL',
        'Berkshire Hathaway': 'BRK'
    }

    company_pattern = re.compile(
        r'\b(?:' + '|'.join(fr'{re.escape(c)}(?:\'s|s)?' for c in companies) + r'|'
        + '|'.join(re.escape(t) for t in stock_tickers.values()) + r')\b',
        re.IGNORECASE
    )
    # Match explicit year ranges or individual years
    year_pattern = re.compile(r'\b(20[0-9]{2})(?:[-\s]*(?:to|through|-)\s*(20[0-9]{2}))?\b', re.IGNORECASE)
    # Match quarters
    quarter_pattern = re.compile(r'\bQ[1-4]\b', re.IGNORECASE)

    # Extract matches
    matched_companies = company_pattern.findall(query)
    matched_years = year_pattern.findall(query)
    matched_quarters = quarter_pattern.findall(query)

    # Normalize company names by matching against the stock tickers
    normalized_companies = []
    for match in matched_companies:
        # Check if match is a ticker, convert to company name
        match = re.sub(r'(\'s|s\')$', '', match, flags=re.IGNORECASE)  # Remove possessive suffix
        for company, ticker in stock_tickers.items():
            if match.upper() == ticker:
                normalized_companies.append(company)
                break
        else:
            # If not a ticker, it's already a company name
            normalized_companies.append(match)

    # Process years
    all_years = set()
    for start, end in matched_years:
        if end:  # If there's a range (e.g., "2022-2024")
            all_years.update(range(int(start), int(end) + 1))
        else:  # Single year
            all_years.add(int(start))

    # Remove duplicates and normalize casing
    normalized_companies = list(set([c.title() for c in normalized_companies]))

    return {
        'Companies': normalized_companies,
        'Years': sorted(all_years),
        'Quarters': [q.upper() for q in matched_quarters]
    }

def retrieval_step(message = "", n = 5, hybrid_search = True, chunk_filtering = False, debug = False, verbose = False):
    """Function to perform the retrieval step of the RAG pipeline. 

    Args:
        message (str): Query to retrieve chunks for. Defaults to "".
        n (int): number of chunks to return. Defaults to 5.
        hybrid_search (bool, optional): Flag to enable or disable hybrid search. Defaults to False.
        debug (bool, optional): Flag to print debugging and other print statements to command line. Defaults to False.
        verbose (bool, optional): Flag to print verbose options of retrieve_n(). Defaults to False.

    Returns:
        ret list(str): top n chunks in list format
    """
    
    # # Define the prompt to use when extracting out the information from the user query
    # extraction_prompt = f"""
    # I am going to give you a users query that they are prompting to a financial chatbot. Take that query, and extract out the company
    # or companies, year or years, and quarter or quearters that relate to the user's query. Then provide that information in three python lists in your response.
    
    # Below is two examples of what your response should look like given an query.
    
    # ### Example Query 1
    
    # Query Structure:
    # ```
    # **Query**: <query>
    #             What is Apple's earnings in 2024?
    #             </query>
    # ```
    
    # Your Response:
    # ```
    # **Companies** = ['Apple']
    # **Years** = [2024]
    # **Quarters** = []
    # ```
    
    # ### Example Query 2
    
    # Query Structure:
    # ```
    # **Query**: <query>
    #             Compare the average revenue of Tesla to Microsoft over the years 2022-2024
    #             </query>
    # ```
    
    # Your Response:
    # ```
    # **Companies** = ['Tesla', 'Microsoft']
    # **Years** = [2022, 2023, 2024]
    # **Quarters** = []
    # ```
    
    # ### Example Query 3
    
    # Query Structure:
    # ```
    # **Query**: <query>
    #             How did Tesla perform during Q2 of 2024?
    #             </query>
    # ```
    
    # Your Response:
    # ```
    # **Companies** = ['Tesla']
    # **Years** = [2024]
    # **Quarters** = ['Q2']
    # ```
    
    # Here is the real user query I would like for you to handle as described above:
    
    # **Query**: <query>
    #             {message}
    #             </query>
    # """
    
    # # get response from Ollama
    # extraction = ollama.generate(model='llama3.1', prompt=extraction_prompt)
    # if debug: print(f"Extraction prompt response: {extraction['response']}")
    # # Use regular expression and the ast library to get the information and turn it into lists
    # companies_match = re.search(r"\*\*Companies\*\*\s*=\s*(\[.*?\])", extraction['response'])
    # years_match = re.search(r"\*\*Years\*\*\s*=\s*(\[.*?\])", extraction['response'])
    # quarters_match = re.search(r"\*\*Quarters\*\*\s*=\s*(\[.*?\])", extraction['response'])
    query_details = extract_query_details(message)
    

    companies_list = query_details["Companies"]
    years_list = query_details["Years"]
    quarters_list = query_details["Quarters"]
    if debug:
        print(companies_list)
        print(years_list)
        print(quarters_list)
    
    # Perform Retrieval and get top n chunks
    return retrieve_n(message, n, companies_list, years_list, quarters_list, hybrid_search=hybrid_search, chunk_filter=chunk_filtering, verbose = verbose)

def generation_step(message = "", top_n = None, eval = False, debug = False):
    """Function to perform the generation step of the RAG pipeline. 

    Args:
        message (str): message for the LLM to answer. Defaults to "".
        top_n (list): List of chunks returned by the retrieval step. 
        debug (bool, optional): Flag to print debugging and other print statements to command line. Defaults to False.

    Returns:
        None
    """
    
    # Build context
    context = ""
    for i, doc in enumerate(top_n):
        context += f"Document {i}: {doc}\n"
    
    
    injected_query = f'''
                    **Context**: <context>
                                {context}
                                <context>
                    **Query**: <query>
                                {message}
                                </query>
                    
                    '''
    
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": injected_query},
    ]

    # Send the chat to the model with streaming enabled
    if eval:
        response = client.chat(model='llama3.1', messages=conversation, stream=False)
        return response.get('message', None).get('content', None)
    else:
        response_stream = client.chat(model='llama3.1', messages=conversation, stream=True)
        print("System: ", end='')
        # Print each chunk of content as it is received
        for chunk in response_stream:
            content = chunk.get('message', {}).get('content', '')
            print(content, end='')
        # Ensure the print ends with a new line
        print("\n")
        
def clear_retrieved_documents(debug = False):
    with open(retrieved_files_path, "w") as file:
        file.write("")  
    if debug: print(f"The file \"{retrieved_files_path}\" has been cleared.")

        
def save_retrieved_documents(documents, query, query_count, debug = False):
    with open(retrieved_files_path, "a") as file:
        file.write("-"*128)
        file.write(f"\nUser Query #{query_count+1}\n")
        file.write("-"*128)
        file.write("\n\n")
        file.write(f"Query: {query}")
        file.write("\n\n")
        file.write("-"*128)
        file.write(f"\nRetrieved Documents\n")
        file.write("-"*128)
        file.write("\n"*2)
        for i, doc in enumerate(documents):
            file.write(f"Document {i}:\n {doc[0]}\n\n\n")
        file.write("_"*128)
        file.write("\n"*3)
        


def run_llm(n = 5, debug = False):
    """Function to combine the retrieval and generation steps into one function. Does so in a way that presents an interactable CLI
    interface in the form of a while loop of user input and llm response.

    Args:
        n (int): Total number of chunks to retrieve on retrieval. Defaults to 5.
        debug (bool, optional): Flag to print debugging and other print statements to command line. Defaults to False.

    Returns:
        None
    """
    
    # Establish connection
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        if(debug): print("Connection established successfully!")

        # Create a cursor to perform database operations
        cursor = conn.cursor()
        
        # Clear retrieved files log
        clear_retrieved_documents()
        query_count = 0
        
        # Welcome user
        print("\nSystem: Welcome to my Financial LLM. Please ask a Question:")
        print()

        # LLM loop
        # Process:
        # - First grab the users query, and ask the model to extract the year and company of focus
        # - Make sure the LLM returns it in an output we can easily extract 
        # - Run the LLM output through a parser to extract that information
        # - use the extracted information to perform Retrieval on documents falling under the year and company
        # - lastly pass the retrieved info to LLM as context along with original query
        # NOTES:
        # - Make sure that the information extract components and RAG enhanced queries are two separate conversations
        while(True):
            message = input("User: ")
            print()
            if(message.lower() == 'exit'):
                cursor.close()
                conn.close()
                break
            
            top_n = retrieval_step(message = message, n = n, hybrid_search=True, chunk_filtering=True)
            
            if debug: print(top_n)
            
            generation_step(message, top_n)
            
            save_retrieved_documents(top_n, message, query_count)
            
            print(f"The retrieved documents have been saved to \"{retrieved_files_path}\" for your review.\n")
            
            query_count += 1
    except Exception as error:
        print(f"Error connecting to the database: {error}")
        
if __name__ == "__main__":
    run_llm(n = 3, debug = False)