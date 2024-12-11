from llm import retrieval_step

sample_query = "What kind of legal proceedings is apple involved in in 2024??"

# retrieve_n(query=sample_query, n = 10, verbose=True)
top_n_chunks = retrieval_step(message = sample_query, n = 10, hybrid_search=True, debug=False, verbose=True)