import json
import psycopg2

# Connection parameters for pgvector
host = "localhost"         # Hostname (since you've mapped the container port)
port = "5432"              # Port you mapped (5432)
dbname = "vectordb"     # Name of your database
user = "admin"           # Username you provided during setup
password = "adminpass"   # Password you provided during setup

def read_and_print_json(file_path):
    try:
        # Open the JSON file and load the data
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Print the contents of the JSON
        # print(json.dumps(data, indent=4))
        print(data[0]['ground_truth'])
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON in file '{file_path}'.")

# Test the function with a sample file path
if __name__ == "__main__":
    file_path = 'eval_dataset.json'  # Replace with your JSON file path
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        line = "The following table shows share-based compensation expense and the related income tax benefit included in the Condensed Consolidated Statements of Operations for the three- and nine-month periods ended June 29, 2024 and July 1, 2023 (in millions)"
        database_query = f"SELECT * FROM text_chunks where text LIKE '%{line}%' LIMIT 5;"
        cursor = conn.cursor()
        # print(database_query)
        # Query data from the table
        cursor.execute(database_query)
        rows = cursor.fetchall()
        print(rows)
    except Exception as error:
        print(f"Error connecting to the database: {error}")

    # read_and_print_json(file_path)
