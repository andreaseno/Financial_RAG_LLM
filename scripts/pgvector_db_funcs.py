import psycopg2
from embedding_gen import generate_embedding

# Connection parameters
host = "localhost"         # Hostname (since you've mapped the container port)
port = "5432"              # Port you mapped (5432)
dbname = "vectordb"     # Name of your database
user = "admin"           # Username you provided during setup
password = "adminpass"   # Password you provided during setup

def retrieve_n(query = "", n = 5, company_filter = [], year_filter = [], verbose = False):
    """Function to retrieve the top n related chunks from the pgvector database using cosine similarity

    Args:
        query (str, optional): Query to retrieve chunks for. Defaults to "".
        n (int, optional): number of chunks to return. Defaults to 5.
        company_filter (list, optional): list of strings where each string is a company identified from the query. Defaults to [].
        year_filter (list, optional): list of ints where each int is a year identified from the query. Defaults to [].
        verbose (bool, optional): Flag to print debugging and other print statements to command line. Defaults to False.

    Returns:
        ret list(str): top n chunks in list format
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
        # print("Connection established successfully!")

        # Create a cursor to perform database operations
        cursor = conn.cursor()

        # Check whether the query is empty
        if(len(query) == 0):
            print("Empty query during retrieval")
            raise 
        
        embedding = generate_embedding(text=query)
        embedding_string = '[' + ','.join(map(str, embedding[0])) + ']'
        
        if (company_filter and year_filter):
            company_filter_string = ', '.join(f"'{company}'" for company in company_filter)
            year_filter_string = ', '.join(f"'{str(year)}'" for year in year_filter)
            database_query = f"SELECT id FROM embedding_chunks WHERE company in ({company_filter_string}) AND year in ({year_filter_string}) ORDER BY embedding <=> '{embedding_string}' LIMIT {n};"
        
        elif (company_filter and not year_filter):
            company_filter_string = ', '.join(f"'{company}'" for company in company_filter)
            database_query = f"SELECT id FROM embedding_chunks WHERE company in ({company_filter_string}) ORDER BY embedding <=> '{embedding_string}' LIMIT {n};"
        
        elif (not company_filter and year_filter):
            year_filter_string = ', '.join(f"'{str(year)}'" for year in year_filter)
            database_query = f"SELECT id FROM embedding_chunks WHERE year in ({year_filter_string}) ORDER BY embedding <=> '{embedding_string}' LIMIT {n};"
            
        else:
            database_query = f"SELECT id FROM embedding_chunks ORDER BY embedding <=> '{embedding_string}' LIMIT {n};"

        # print(database_query)
        # Query data from the table
        cursor.execute(database_query)
        rows = cursor.fetchall()

        ret = []
        
        # Print query results
        for row in rows:
            # Query data from the table
            cursor.execute(f"SELECT text FROM text_chunks where id = {row[0]};")
            text = cursor.fetchone()
            if verbose:
                print(row)
                print(text, end="\n\n\n\n\n")
            ret.append(text)
            
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        return ret

    except Exception as error:
        print(f"Error connecting to the database: {error}")
        
# retrieve_n(verbose=True)