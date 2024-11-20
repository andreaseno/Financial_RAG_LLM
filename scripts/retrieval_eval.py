import json
import psycopg2
from llm import retrieval_step

# Global Variables
debug = True
k = 5

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
    
    sum_precision_k = 0
    sum_recall_k = 0
    sum_f1_k = 0
    
    company_averages = {}
    keyword_averages = {}
    
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
            for j, chunk in enumerate(top_n_chunks):
                print(f"Chunk {j+1}/{k}:\n {chunk[0]}")
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
        if (precision_k+recall_k > 0):
            f1_k = (2*precision_k*recall_k)/(precision_k+recall_k)
        else:
            f1_k = 0
        print(f"Entry #{i} has Precision@k: {precision_k}")
        print(f"Entry #{i} has Recall@k: {recall_k}")
        print(f"Entry #{i} has F1@k: {f1_k}")
        
        sum_precision_k += precision_k
        sum_recall_k += recall_k
        sum_f1_k += f1_k
        
        for company in entry["companies"]:
            score_dict = company_averages.get(company, False)
            if(score_dict):
                company_averages[company]["precision_k"] +=  precision_k
                company_averages[company]["recall_k"] +=  recall_k
                company_averages[company]["f1_k"] +=  f1_k
                company_averages[company]["count"] += 1
            else:
                company_averages[company] = {
                    "precision_k": precision_k,
                    "recall_k": recall_k,
                    "f1_k":  f1_k,
                    "count": 1
                }
        for keyword in entry["keywords"]:
            score_dict = keyword_averages.get(keyword, False)
            if(score_dict):
                keyword_averages[keyword]["precision_k"] +=  precision_k
                keyword_averages[keyword]["recall_k"] +=  recall_k
                keyword_averages[keyword]["f1_k"] +=  f1_k
                keyword_averages[keyword]["count"] += 1
            else:
                keyword_averages[keyword] = {
                    "precision_k": precision_k,
                    "recall_k": recall_k,
                    "f1_k":  f1_k,
                    "count": 1
                }

    
    avg_precision_k = sum_precision_k/len(data)
    avg_recall_k = sum_recall_k/len(data)
    avg_f1_k = sum_f1_k/len(data)
    
    print("\n\n\n")
    print("-"*128)
    print(f"Average Precision@k: {avg_precision_k}")
    print(f"Average Recall@k: {avg_recall_k}")
    print(f"Average F1@k: {avg_f1_k}")
    print("\n\n\n")
    print("Company Averages:")
    for company, scores in company_averages.items():
        avg_precision = scores["precision_k"] / scores["count"]
        avg_recall = scores["recall_k"] / scores["count"]
        avg_f1 = scores["f1_k"] / scores["count"]
        print(f"Company: {company}, Precision@k: {avg_precision:.2f}, Recall@k: {avg_recall:.2f}, F1@k: {avg_f1:.2f}")

    print("\nKeyword Averages:")
    for keyword, scores in keyword_averages.items():
        avg_precision = scores["precision_k"] / scores["count"]
        avg_recall = scores["recall_k"] / scores["count"]
        avg_f1 = scores["f1_k"] / scores["count"]
        print(f"Keyword: {keyword}, Precision@k: {avg_precision:.2f}, Recall@k: {avg_recall:.2f}, F1@k: {avg_f1:.2f}")
    print("\n\n\n")
        

            
        
        

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode the JSON in file '{file_path}'.")
