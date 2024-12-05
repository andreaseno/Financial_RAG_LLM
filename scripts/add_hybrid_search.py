import psycopg2
import traceback

# set globals
debug = False

# Connection parameters for pgvector
host = "localhost"         # Hostname (since you've mapped the container port)
port = "5432"              # Port you mapped (5432)
dbname = "vectordb"     # Name of your database
user = "admin"           # Username you provided during setup
password = "adminpass"   # Password you provided during setup

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
    
    # Ensure the `pg_trgm` extension is enabled (for indexing)
    cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    conn.commit()
    print("pg_trgm extension ensured.")

    # Check and ensure the `text_chunks` table exists
    cursor.execute("""
        SELECT to_regclass('public.text_chunks');
    """)
    result = cursor.fetchone()
    if not result[0]:
        raise Exception("Table 'text_chunks' does not exist. Create the table before proceeding.")
    else: print("text_chunks table does exist")

    # Add the `text_vectors` column to store `tsvector`
    cursor.execute("""
        ALTER TABLE text_chunks
        ADD COLUMN IF NOT EXISTS text_vectors tsvector;
    """)
    conn.commit()
    print("Column `text_vectors` added or already exists.")

    # Populate the `text_vectors` column using the `to_tsvector` function
    cursor.execute("""
        UPDATE text_chunks
        SET text_vectors = to_tsvector('english', text);
    """)
    conn.commit()
    print("Column `text_vectors` populated with tsvector data.")

    # Index the `text_vectors` column
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS text_vectors_idx
        ON text_chunks
        USING gin(text_vectors);
    """)
    conn.commit()
    print("Index on `text_vectors` column created successfully.")
    
    

except Exception as error:
    tb = traceback.format_exc()
    print(f"Error connecting to the database: {error}")
    print(tb)
    
finally:
    if 'cursor' in locals() and cursor:
        cursor.close()
    if 'conn' in locals() and conn:
        conn.close()
