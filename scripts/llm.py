import ollama
import json
import psycopg2
import ast
import re
from pgvector_db_funcs import retrieve_n


# set globals
debug = False

# Connection parameters for pgvector
host = "localhost"         # Hostname (since you've mapped the container port)
port = "5432"              # Port you mapped (5432)
dbname = "vectordb"     # Name of your database
user = "admin"           # Username you provided during setup
password = "adminpass"   # Password you provided during setup

# Initialize the Ollama client
client = ollama.Client()
def run_llm(eval = False, n = 5, eval_query = None):
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
        if (not eval): 
            print("System: Welcome to my Financial LLM. Please ask a Question:")
            print()
                        
        # # Open the .md file and read its content
        # with open('../md_files/_10-Q-Q3-2024-As-Filed-pages1-2.pdf.md', 'r', encoding='utf-8') as file:
        #     md_content = file.read()
        
        
    #     md_content = """# Apple Inc.

    # ## CONDENSED CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME (Unaudited)
    # ### (In millions)

    # | Description                                                                                          | Three Months Ended |         | Nine Months Ended |         |
    # |------------------------------------------------------------------------------------------------------|---------------------|---------|-------------------|---------|
    # |                                                                                                      | July 1, 2023       | June 25, 2022 | July 1, 2023     | June 25, 2022 |
    # | Net income                                                                                           | $ 19,881            | $ 19,442 | $ 74,039          | $ 79,082 |
    # | Other comprehensive income/(loss):                                                                    |                     |         |                   |         |
    # | Change in foreign currency translation, net of tax                                                  | (385)               | (721)   | (494)             | (1,102) |
    # | Change in unrealized gains/losses on derivative instruments, net of tax:                           |                     |         |                   |         |
    # | Change in fair value of derivative instruments                                                       | 509                 | 852     | (492)             | 1,548   |
    # | Adjustment for net (gains)/losses realized and included in net income                              | 103                 | 121     | (1,854)           | (87)    |
    # | Total change in unrealized gains/losses on derivative instruments                                   | 612                 | 973     | (2,346)           | 1,461   |
    # | Change in unrealized gains/losses on marketable debt securities, net of tax:                      |                     |         |                   |         |
    # | Change in fair value of marketable debt securities                                                  | (340)               | (3,150) | 1,963             | (9,959) |
    # | Adjustment for net (gains)/losses realized and included in net income                              | 58                  | 95      | 185               | 140     |
    # | Total change in unrealized gains/losses on marketable debt securities                               | (282)               | (3,055) | 2,148             | (9,819) |
    # | 
    # """
            
        # system_prompt = f"""You are an AI assistant tasked with answering financial questions about Apple. Your task is to answer simple questions about the company Apple based on the information provided as context.
        #                     Here is a snippet from the most recent 10Q for Apple document in markdown format as context. 
        #                     <apple_10Q>
        #                     {md_content}
        #                     </apple_10Q>
        #                     When the user asks a question about Apple, reference back to this document to answer the question.
                            
        #                     Your model has been trained on knowledge up to approximately March or April 2023.
                            
                            
                            
                            
        #                 """
        
        system_prompt = f"""You are an AI assistant tasked with answering financial questions. Your task is to answer simple questions about a company based on the <context> element of the query.
                            Here is an example query in the same format queries will be asked
                            
                            ```
                            **Context**: <context>
                                        ...
                                        <context>
                            **Query**: <query>
                                        ...
                                        </query>
                            ```
                            
                            When the user asks a question about a company, use information from the <context> element to answer the question asked in <query>.
                            
                            
                            
                            
                        """

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
            if (not eval): 
                message = input("User: ")
                print()
                if(message.lower() == 'exit'):
                    cursor.close()
                    conn.close()
                    break
            else:
                message = eval_query
            
            # Define the prompt to use when extracting out the information from the user query
            extraction_prompt = f"""
            I am going to give you a users query that they are prompting to a financial chatbot. Take that query, and extract out the company
            or companies, and year or years that relate to the user's query. Then provide that information in two python lists in your response.
            
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
            **Companies** = ["Tesla", "Microsoft"]
            **Years** = [2022, 2023, 2024]
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
            companies_list = ast.literal_eval(companies_match.group(1)) if companies_match else []
            years_list = ast.literal_eval(years_match.group(1)) if years_match else []
            if debug:
                print(companies_list)
                print(years_list)
            
            # Perform Retrieval and get top n chunks
            top_n = retrieve_n(message, n, companies_list, years_list)
            
            if debug: print(top_n)
            if (not eval):
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
                response_stream = client.chat(model='llama3.1', messages=conversation, stream=True)
                print("System: ", end='')
                # Print each chunk of content as it is received
                for chunk in response_stream:
                    content = chunk.get('message', {}).get('content', '')
                    print(content, end='')
                # Ensure the print ends with a new line
                print("\n")
            else:
                return top_n
            

    except Exception as error:
        print(f"Error connecting to the database: {error}")