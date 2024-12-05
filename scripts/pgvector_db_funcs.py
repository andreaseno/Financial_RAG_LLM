import psycopg2
from embedding_gen import generate_embedding
import re
from sklearn.preprocessing import MinMaxScaler
import pandas as pd



# Connection parameters
host = "localhost"         # Hostname (since you've mapped the container port)
port = "5432"              # Port you mapped (5432)
dbname = "vectordb"     # Name of your database
user = "admin"           # Username you provided during setup
password = "adminpass"   # Password you provided during setup

def retrieve_n(query = "", n = 5, company_filter = [], year_filter = [], quarters_filter = [], hybrid_search = True, verbose = False):
    """Function to retrieve the top n related chunks from the pgvector database using cosine similarity

    Args:
        query (str, optional): Query to retrieve chunks for. Defaults to "".
        n (int, optional): number of chunks to return. Defaults to 5.
        company_filter (list, optional): list of strings where each string is a company identified from the query. Defaults to [].
        year_filter (list, optional): list of ints where each int is a year identified from the query. Defaults to [].
        quarters_filter (list, optional): list of strings where each string is a quarter identified from the query. Defaults to [].
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

        # Create a cursor to perform database operations
        cursor = conn.cursor()

        # Check whether the query is empty
        if(len(query) == 0):
            print("Empty query during retrieval")
            raise 
        
        # STEP 1: Vector search
        embedding = generate_embedding(text=query)
        embedding_string = '[' + ','.join(map(str, embedding[0])) + ']'
        database_query = f"SELECT id, 1 - (embedding <=> '{embedding_string}') AS cosine_similarity FROM embedding_chunks "
        add_and = False
        if (company_filter):
            company_filter_string = ', '.join(f"'{company}'" for company in company_filter)
            # year_filter_string = ', '.join(f"'{str(year)}'" for year in year_filter)
            database_query += f"WHERE company in ({company_filter_string}) "
            add_and = True
        
        if (year_filter):
            if add_and: database_query += "AND "
            else: database_query += "WHERE "
            year_filter_string = ', '.join(f"'{str(year)}'" for year in year_filter)
            database_query += f"year in ({year_filter_string}) "
            add_and = True
            
        if (quarters_filter):
            if add_and: database_query += "AND "
            else: database_query += "WHERE "
            quarters_filter_string = ', '.join(f"'{quarter}'" for quarter in quarters_filter)
            database_query += f"fiscal_quarter in ({quarters_filter_string}) "
            add_and = True

        database_query += f"ORDER BY embedding <=> '{embedding_string}';"

        # if verbose: print(database_query)
        # Query data from the table
        cursor.execute(database_query)
        vector_rows = cursor.fetchall()
        
        # convert to dataframe
        semantic_df = pd.DataFrame(vector_rows, columns=['id', 'cosine_similarity'])
        
        # STEP 2: Full Text Search
        # Define list of stop words to remove from query 
        # if hybrid_search:
        #     stopwords = [
        #         'apple', 
        #         'tesla', 
        #         'nvidia', 
        #         'microsoft', 
        #         'meta', 
        #         'google', 
        #         'berkshire', 
        #         'hathaway', 
        #         'amazon', 
        #         'q1', 
        #         'q2', 
        #         'q3', 
        #         'q4', 
        #         '2025', 
        #         '2024', 
        #         '2023', 
        #         '2022', 
        #         '2021', 
        #         '10q', 
        #         '10k', 
        #         '10q-10k', 
        #         '10q/10k'
        #     ]
            
        #     # Filter out stop words
        #     query_words = re.findall(r'\w+', query.lower())
        #     filtered_words = [word for word in query_words if word not in stopwords]
        #     filtered_query = ' '.join(filtered_words)
            
        #     # Use plainto_tsquery with the filtered query
        #     cursor.execute("""
        #                     SELECT id, text,
        #                         ts_rank(text_vectors, plainto_tsquery('english', %s)) AS rank
        #                     FROM text_chunks
        #                     ORDER BY rank DESC;
        #                 """, (filtered_query,))
        #     full_text_rows = cursor.fetchall()
        
        #     # Convert results to DataFrames
        #     text_df = pd.DataFrame(full_text_rows, columns=['id', 'text', 'full_text_score'])
            
        #     # Merge results on 'id'
        #     scoring_df = pd.merge(semantic_df, text_df, on='id', how='outer').fillna(-1)

        #     # Normalize scores
        #     scaler = MinMaxScaler()
        #     scoring_df[['cosine_similarity', 'full_text_score']] = scaler.fit_transform(
        #         scoring_df[['cosine_similarity', 'full_text_score']]
        #     )
            
        #     # compute combined score
        #     scoring_df['missing_cosine'] = scoring_df['cosine_similarity'] == -1
        #     scoring_df['missing_text'] = scoring_df['full_text_score'] == -1
        #     scoring_df['score'] = scoring_df.apply(
        #         lambda row: row['cosine_similarity'] if row['missing_text'] else
        #                     row['full_text_score'] if row['missing_cosine'] else
        #                     0.5 * row['cosine_similarity'] + 0.5 * row['full_text_score'],
        #         axis=1
        #     )
            
        #     # Sort by combined score
        #     scoring_df = scoring_df.sort_values('score', ascending=False)
        # else: 
        #     scoring_df = semantic_df
        #     scoring_df['score'] = scoring_df['cosine_similarity']
        #     scoring_df = scoring_df.sort_values('score', ascending=False)
    
        stopwords = [
            'apple', 
            'tesla', 
            'nvidia', 
            'microsoft', 
            'meta', 
            'google', 
            'berkshire', 
            'hathaway', 
            'amazon', 
            'q1', 
            'q2', 
            'q3', 
            'q4', 
            '2025', 
            '2024', 
            '2023', 
            '2022', 
            '2021', 
            '10q', 
            '10k', 
            '10q-10k', 
            '10q/10k'
        ]
        
        # Filter out stop words
        query_words = re.findall(r'\w+', query.lower())
        filtered_words = [word for word in query_words if word not in stopwords]
        filtered_query = ' '.join(filtered_words)
        
        # Use plainto_tsquery with the filtered query
        cursor.execute("""
                        SELECT id, text,
                            ts_rank(text_vectors, plainto_tsquery('english', %s)) AS rank
                        FROM text_chunks
                        ORDER BY rank DESC;
                    """, (filtered_query,))
        full_text_rows = cursor.fetchall()
    
        # Convert results to DataFrames
        text_df = pd.DataFrame(full_text_rows, columns=['id', 'text', 'full_text_score'])
        
        # Merge results on 'id'
        scoring_df = pd.merge(semantic_df, text_df, on='id', how='outer').fillna(-1)

        # Normalize scores
        scaler = MinMaxScaler()
        scoring_df[['cosine_similarity', 'full_text_score']] = scaler.fit_transform(
            scoring_df[['cosine_similarity', 'full_text_score']]
        )
        
        # compute combined score
        scoring_df['missing_cosine'] = scoring_df['cosine_similarity'] == -1
        scoring_df['missing_text'] = scoring_df['full_text_score'] == -1
        scoring_df['score'] = scoring_df.apply(
            lambda row: row['cosine_similarity'] if row['missing_text'] else
                        row['full_text_score'] if row['missing_cosine'] else
                        0.80 * row['cosine_similarity'] + 0.2 * row['full_text_score'],
            axis=1
        )
        if hybrid_search:
            # Sort by combined score
            scoring_df = scoring_df.sort_values('score', ascending=False)
        else:
            # Sort by combined score
            scoring_df = scoring_df.sort_values('cosine_similarity', ascending=False)
        # Print or use the results
        print(scoring_df.head(10))

        ret = []
        # Print Hybrid search results
        for index, row in scoring_df.head(n).iterrows():
            cursor.execute(f"SELECT text FROM text_chunks where id = {row['id']};")
            text = cursor.fetchone()
            if verbose:
                print(f"ID: {row['id']}, Combined Score: {row['score']} Semantic Score: {row['cosine_similarity']}")
                print(f"Text: {text}\n")
                print("\n"*3)
            ret.append(text)
        # for row in vector_rows:
        #     # Query data from the table
        #     cursor.execute(f"SELECT text FROM text_chunks where id = {row[0]};")
        #     text = cursor.fetchone()
        #     if verbose:
        #         print(row)
        #         print(text, end="\n\n\n\n\n")
        #     ret.append(text)
            
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        return ret

    except Exception as error:
        print(f"Error connecting to the database: {error}")
        
# retrieve_n(verbose=True)