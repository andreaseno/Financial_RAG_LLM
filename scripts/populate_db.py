from embedding_gen import generate_embedding
from chunking import chunk_markdown, write_debug_log
import psycopg2
import os
import glob
import tiktoken
from transformers import AutoTokenizer, AutoModel
import torch

# Load the tokenizer and the base model (without classification head)
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")

def save_embedding_to_db(embedding, chunk):
    try:
        # Convert the numpy array to a list
        embedding_list = embedding.tolist()[0]  # Flatten the array
        # Create the SQL insert statement
        insert_query = """
        INSERT INTO embedding_chunks (embedding)
        VALUES (%s)
        RETURNING id;
        """

        cursor.execute(insert_query, (embedding_list,))
        embedding_id = cursor.fetchone()[0]  # Fetch the returned id
        conn.commit()
        
        # Now insert the text into markdown_text with the same id
        insert_text_query = """
        INSERT INTO text_chunks (id, text)
        VALUES (%s, %s);
        """

        # Execute the query using the retrieved embedding_id
        cursor.execute(insert_text_query, (embedding_id, chunk))
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

def process_markdown_file(file_path):
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
                save_embedding_to_db(embedding, table)
            elif isinstance(chunk, str):
                embedding = generate_embedding(text=chunk)
                # print(f"{i}, ", end="")
                # Store or process the embedding here
                save_embedding_to_db(embedding, chunk)
        print(end="\n\n")

def read_markdown_files(base_dir):
    """Recursively reads markdown files in all subdirectories."""
    for subdir, dirs, files in os.walk(base_dir):
        for file in glob.glob(os.path.join(subdir, "*.md")):
            print(f"Reading {file}")
            process_markdown_file(file)
           
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
        CREATE TABLE IF NOT EXISTS embedding_chunks (id bigserial PRIMARY KEY, embedding vector(768));

    """)
    conn.commit()  # Commit the transaction
    
    # Example query: Create a table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS text_chunks (id bigint PRIMARY KEY, text text);

    """)
    conn.commit()  # Commit the transaction
    
    # Read and process markdown files
    read_markdown_files(base_dir)
    print("\nFinished populating database")
    

except Exception as error:
    print(f"Error connecting to the database: {error}")


