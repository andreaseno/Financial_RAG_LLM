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

system_prompt = """You are an AI assistant tasked with answering financial questions. Your task is to answer simple questions about a company based on the <context> element of the query.
    Here is an example query in the same format queries will be asked
    
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

def retrieval_step(message = "", n = 5, hybrid_search = True, debug = False):
    """Function to perform the retrieval step of the RAG pipeline. 

    Args:
        message (str): Query to retrieve chunks for. Defaults to "".
        n (int): number of chunks to return. Defaults to 5.
        debug (bool, optional): Flag to print debugging and other print statements to command line. Defaults to False.

    Returns:
        ret list(str): top n chunks in list format
    """
    
    # Define the prompt to use when extracting out the information from the user query
    extraction_prompt = f"""
    I am going to give you a users query that they are prompting to a financial chatbot. Take that query, and extract out the company
    or companies, year or years, and quarter or quearters that relate to the user's query. Then provide that information in three python lists in your response.
    
    Below is two examples of what your response should look like given an query.
    
    ### Example Query 1
    
    Query Structure:
    ```
    **Query**: <query>
                What is Apple's earnings in 2024?
                </query>
    ```
    
    Your Response:
    ```
    **Companies** = ['Apple']
    **Years** = [2024]
    **Quarters** = []
    ```
    
    ### Example Query 2
    
    Query Structure:
    ```
    **Query**: <query>
                Compare the average revenue of Tesla to Microsoft over the years 2022-2024
                </query>
    ```
    
    Your Response:
    ```
    **Companies** = ['Tesla', 'Microsoft']
    **Years** = [2022, 2023, 2024]
    **Quarters** = []
    ```
    
    ### Example Query 3
    
    Query Structure:
    ```
    **Query**: <query>
                How did Tesla perform during Q2 of 2024?
                </query>
    ```
    
    Your Response:
    ```
    **Companies** = ['Tesla']
    **Years** = [2024]
    **Quarters** = ['Q2']
    ```
    
    Here is the real user query I would like for you to handle as described above:
    
    **Query**: <query>
                {message}
                </query>
    """
    
    # get response from Ollama
    extraction = ollama.generate(model='llama3.1', prompt=extraction_prompt)
    if debug: print(f"Extraction prompt response: {extraction['response']}")
    # Use regular expression and the ast library to get the information and turn it into lists
    companies_match = re.search(r"\*\*Companies\*\*\s*=\s*(\[.*?\])", extraction['response'])
    years_match = re.search(r"\*\*Years\*\*\s*=\s*(\[.*?\])", extraction['response'])
    quarters_match = re.search(r"\*\*Quarters\*\*\s*=\s*(\[.*?\])", extraction['response'])
    companies_list = ast.literal_eval(companies_match.group(1)) if companies_match else []
    years_list = ast.literal_eval(years_match.group(1)) if years_match else []
    quarters_list = ast.literal_eval(quarters_match.group(1)) if companies_match else []
    if debug:
        print(companies_list)
        print(years_list)
        print(quarters_list)
    
    # Perform Retrieval and get top n chunks
    return retrieve_n(message, n, companies_list, years_list, quarters_list, hybrid_search=hybrid_search)

def generation_step(message = "", top_n = None, conversation = None, eval = False, debug = False):
    """Function to perform the generation step of the RAG pipeline. 

    Args:
        message (str): message for the LLM to answer. Defaults to "".
        top_n (list): List of chunks returned by the retrieval step. 
        conversation (list): Conversation that was created before user was queried. Should contain all previous questions asked. 
        debug (bool, optional): Flag to print debugging and other print statements to command line. Defaults to False.

    Returns:
        None
    """
    
    if (not conversation):
        conversation = [
            {"role": "system", "content": system_prompt},
        ]
    
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
    
    conversation.append({"role": "user", "content": injected_query})

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
        
        # Welcome user
        print("System: Welcome to my Financial LLM. Please ask a Question:")
        print()

        # Define conversation
        conversation = [
            {"role": "system", "content": system_prompt},
        ]
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
            
            top_n = retrieval_step(message = message, n = n, hybrid_search=False)
            
            if debug: print(top_n)
            
            generation_step(message, top_n, conversation)
            
    except Exception as error:
        print(f"Error connecting to the database: {error}")
        
if __name__ == "__main__":
    run_llm(n = 3, debug = True)