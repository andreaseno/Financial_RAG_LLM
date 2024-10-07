import psycopg2
from embedding_gen import generate_embedding

# Connection parameters
host = "localhost"         # Hostname (since you've mapped the container port)
port = "5432"              # Port you mapped (5432)
dbname = "vectordb"     # Name of your database
user = "admin"           # Username you provided during setup
password = "adminpass"   # Password you provided during setup

def retrieve_n(query = "", n = 5, verbose = False):
    # Establish connection
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        print("Connection established successfully!")

        # Create a cursor to perform database operations
        cursor = conn.cursor()

        # Check whether the query is empty
        if(len(query) == 0):
            print("Empty query during retrieval")
            raise 
        
        embedding = generate_embedding(text=query)
        embedding_string = '[' + ','.join(map(str, embedding[0])) + ']'

        # Query data from the table
        cursor.execute(f"SELECT id FROM embedding_chunks ORDER BY embedding <=> '{embedding_string}' LIMIT {n};")
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
            
        return ret
        # Close the cursor and connection
        cursor.close()
        conn.close()

    except Exception as error:
        print(f"Error connecting to the database: {error}")
        
# retrieve_n(verbose=True)