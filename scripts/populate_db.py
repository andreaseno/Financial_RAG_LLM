from embedding_gen import generate_embedding
from chunking import chunk_markdown, write_debug_log
import psycopg2
import os
import glob
import tiktoken
from transformers import AutoTokenizer, AutoModel
import torch
import traceback


# Load the tokenizer and the base model (without classification head)
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")

def save_embedding_to_db(embedding, chunk, company_name, doc_year, doc_type, fiscal_quarter):
    try:
        # Convert the numpy array to a list
        embedding_list = embedding.tolist()[0]  # Flatten the array
        # Create the SQL insert statement
        insert_query = """
        INSERT INTO embedding_chunks (embedding, company, year, document_type, fiscal_quarter)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """

        cursor.execute(insert_query, (embedding_list, company_name, str(doc_year), doc_type, fiscal_quarter))
        embedding_id = cursor.fetchone()[0]  # Fetch the returned id
        conn.commit()
        
        # Now insert the text into markdown_text with the same id
        insert_text_query = """
        INSERT INTO text_chunks (id, text, company, year, document_type, fiscal_quarter)
        VALUES (%s, %s, %s, %s, %s, %s);
        """

        # Execute the query using the retrieved embedding_id
        cursor.execute(insert_text_query, (embedding_id, chunk, company_name, str(doc_year), doc_type, fiscal_quarter))
        conn.commit()
    except Exception as error:
        print(f"ERROR committing to the database: {error}")
        print(f"chunk text: {chunk}")
        write_debug_log(f"ERROR committing to the database: {error}")
        write_debug_log(f"chunk text: {chunk}")
# # Function to split text into chunks based on token size
# def split_into_chunks(text, max_tokens=200):
#     """Splits the input text into chunks of a specified token size."""
#     tokenizer = tiktoken.get_encoding("cl100k_base")  # Change to the tokenizer you are using
#     tokens = tokenizer.encode(text)
    
#     # Chunk tokens into groups of max_tokens size
#     for i in range(0, len(tokens), max_tokens):
#         yield tokenizer.decode(tokens[i:i + max_tokens])

def process_markdown_file(file_path, company_name, doc_year, doc_type, fiscal_quarter):
    """Function to process each markdown file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # print("Chunk #: ", end="")
        write_debug_log(f"Chunking {file}")
        chunks = chunk_markdown(content,512, verbose=False)
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, tuple):
                table, summary = chunk
                embedding = generate_embedding(text=summary)
                # print(f"{i}, ", end="")
                # Store or process the embedding here
                save_embedding_to_db(embedding, table, company_name, doc_year, doc_type, fiscal_quarter)
            elif isinstance(chunk, str):
                embedding = generate_embedding(text=chunk)
                # print(f"{i}, ", end="")
                # Store or process the embedding here
                save_embedding_to_db(embedding, chunk, company_name, doc_year, doc_type, fiscal_quarter)
        print(end="\n\n")

def read_markdown_files(base_dir):
    """Recursively reads markdown files in all subdirectories."""
    for subdir, dirs, files in os.walk(base_dir):
        for file in glob.glob(os.path.join(subdir, "*.md")):
            file_parts = file.split('/')
            company = file_parts[2]
            year = file_parts[3]
            name_info = file_parts[5].split('-')
            doc_type = name_info[0]
            fiscal_quarter = name_info[1]
            print(f"Reading {file}")
            # print(f"company: {company} year: {year} file type: {doc_type}")
            process_markdown_file(file, company, year, doc_type, fiscal_quarter)
           
# Define the path to the folder with markdown files
base_dir = "../md_files"

# set globals
debug = False

# Connection parameters for pgvector
host = "localhost"         # Hostname (since you've mapped the container port)
port = "5432"              # Port you mapped (5432)
dbname = "vectordb"     # Name of your database
user = "admin"           # Username you provided during setup
password = "adminpass"   # Password you provided during setup

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
    
    # Example query: Create a table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_chunks (id bigserial PRIMARY KEY, embedding vector(768), company TEXT, year VARCHAR(4), document_type VARCHAR(255), fiscal_quarter VARCHAR(2));

    """)
    conn.commit()  # Commit the transaction
    
    # Example query: Create a table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS text_chunks (id bigint PRIMARY KEY, text TEXT, company TEXT, year VARCHAR(4), document_type VARCHAR(255), fiscal_quarter VARCHAR(2));

    """)
    conn.commit()  # Commit the transaction
    
    # Read and process markdown files
    read_markdown_files(base_dir)
    
    # Delete any duplicate chunks
    cursor.execute("""DELETE FROM text_chunks
                        WHERE id IN (
                            SELECT id
                            FROM (
                                SELECT 
                                    id,
                                    ROW_NUMBER() OVER (
                                        PARTITION BY text, company, year, document_type, fiscal_quarter 
                                        ORDER BY id
                                    ) AS row_num
                                FROM text_chunks
                            ) AS subquery
                            WHERE row_num > 1
                        );
                    DELETE FROM embedding_chunks
                        WHERE id NOT IN (SELECT id FROM text_chunks);
                    DELETE FROM embedding_chunks
                        WHERE id IN (
                            SELECT id
                            FROM (
                                SELECT 
                                    id,
                                    ROW_NUMBER() OVER (
                                        PARTITION BY embedding, company, year, document_type, fiscal_quarter 
                                        ORDER BY id
                                    ) AS row_num
                                FROM embedding_chunks
                            ) AS subquery
                            WHERE row_num > 1
                        );
                    DELETE FROM text_chunks
                        WHERE id NOT IN (SELECT id FROM embedding_chunks);
                        """)
    conn.commit()
    print("\nFinished populating database")
    

except Exception as error:
    tb = traceback.format_exc()
    print(f"Error connecting to the database: {error}")
    print(tb)




