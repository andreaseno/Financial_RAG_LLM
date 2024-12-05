from pgvector_db_funcs import retrieve_n

sample_query = "What was Apple's revenue in Q3 of 2024?"

retrieve_n(query=sample_query, n = 10, verbose=True)