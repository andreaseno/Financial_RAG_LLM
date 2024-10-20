from transformers import AutoTokenizer, AutoModel
from funcs import write_debug_log
import torch

# Load the tokenizer and the base model (without classification head)
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")

MAX_CHUNK_SIZE = 512

def generate_embedding(tokenizer=tokenizer, model=model, text="", max_chunk_size=MAX_CHUNK_SIZE):
    # Tokenize the input text and return PyTorch tensors
    encoded_input = tokenizer(text, max_length=512, padding=False, return_tensors='pt')

    # Pass the encoded input to the model to get hidden states
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(**encoded_input)

    # Get the last hidden state (this is the embeddings for each token)
    last_hidden_state = outputs.last_hidden_state

    # Use the [CLS] token's embedding as the sentence embedding (first token's output)
    cls_embedding = last_hidden_state[:, 0, :]

    # Convert to a numpy array if needed
    cls_embedding_np = cls_embedding.cpu().numpy()
    
    # if (cls_embedding_np.shape[1] > max_chunk_size):
    #     write_debug_log(message=f"ERROR IN embedding_gen.py. Embedding created with size {cls_embedding_np.shape[1]} and max size is {max_chunk_size}. The raw text is here:")
    #     write_debug_log(message=text)
    
    return cls_embedding_np


# # Example text to convert into embeddings
# text = """# Apple Inc.

# ## CONDENSED CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME (Unaudited)
# ### (In millions)

# | Description                                                                                          | Three Months Ended |         | Nine Months Ended |         |
# |------------------------------------------------------------------------------------------------------|---------------------|---------|-------------------|---------|
# |                                                                                                      | July 1, 2023       | June 25, 2022 | July 1, 2023     | June 25, 2022 |
# | Net income                                                                                           | $ 19,881            | $ 19,442 | $ 74,039          | $ 79,082 |
# | Other comprehensive income/(loss):                                                                    |                     |         |                   |         |
# | Change in foreign currency translation, net of tax                                                  | (385)               | (721)   | (494)             | (1,102) |
# | Change in unrealized gains/losses on derivative instruments, net of tax:                           |                     |         |                   |         |
# | Change in fair value of derivative instruments                                                       | 509                 | 852     | (492)             | 1,548   |
# | Adjustment for net (gains)/losses realized and included in net income                              | 103                 | 121     | (1,854)           | (87)    |
# | Total change in unrealized gains/losses on derivative instruments                                   | 612                 | 973     | (2,346)           | 1,461   |
# | Change in unrealized gains/losses on marketable debt securities, net of tax:                      |                     |         |                   |         |
# | Change in fair value of marketable debt securities                                                  | (340)               | (3,150) | 1,963             | (9,959) |
# | Adjustment for net (gains)/losses realized and included in net income                              | 58                  | 95      | 185               | 140     |
# | Total change in unrealized gains/losses on marketable debt securities                               | (282)               | (3,055) | 2,148             | (9,819) |
# | 
# """

# # Convert to a numpy array if needed
# cls_embedding_np = generate_embedding(tokenizer, model, text)

# # Print or save the embedding
# print(cls_embedding_np)
# print(f"Embedding np size: {len(cls_embedding_np[0])}")
