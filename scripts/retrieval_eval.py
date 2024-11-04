import json
import psycopg2
from llm import retrieval_step

# Global Variables
debug = True
k = 3

# Connection parameters for pgvector
host = "localhost"         # Hostname (since you've mapped the container port)
port = "5432"              # Port you mapped (5432)
dbname = "vectordb"     # Name of your database
user = "admin"           # Username you provided during setup
password = "adminpass"   # Password you provided during setup

file_path = 'eval_dataset.json'

try:
    # Open the JSON file and load the data
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    print(f"\nEVALUATION OF TOP {k} RETRIEVAL:\n")
    
    # Loop through entries in the eval dataset
    for i, entry in enumerate(data):
        query = entry['query']
        if debug: 
            print("\n")
            print("-"*128)
            print("\n")
            print(f"Query: {query}\n")
        # Grab the top n chunks using the same RAG algorithm used in main.py
        # top_n_chunks = run_llm(eval = True, n = 50, eval_query = query)
        top_n_chunks = retrieval_step(message = query, n = k)
        if debug:
            for i, chunk in enumerate(top_n_chunks):
                print(f"Chunk {i+1}/{k}:\n {chunk[0]}")
                print()
            print()
        true_positives = 0
        # For each chunk in top_n_chunks, check whether it is a true or false positive using test dataset
        for chunk in top_n_chunks:
            for ground_truth in entry["ground_truth"]:
                # For debugging purposes
                # print(f"chunk: {chunk[0]}\n\n")
                # print(f"Ground truth: {ground_truth['text']}\n\n")
                if ground_truth["text"] in chunk[0]:
                    # print("Found a match!")
                    true_positives += 1
                    break
        # Calculate Precision@k and Recall@k where k = n
        precision_k = true_positives/k
        recall_k = true_positives/len(entry["ground_truth"])
        print(f"Entry #{i} has Precision@k: {precision_k}")
        print(f"Entry #{i} has Recall@k: {recall_k}")
        

            
        
        

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode the JSON in file '{file_path}'.")
