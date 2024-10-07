from chunking import chunk_markdown
from embedding_gen import generate_embedding

test_md = """expected for the full year or for any other future years or interim periods.

### Reclassifications
Certain prior period balances have been reclassified to conform to the current period presentation in the accompanying notes.

### Revenue Recognition
#### Revenue by source
The following table disaggregates our revenue by major source (in millions):

| Source                                          | Three Months Ended September 30, | Nine Months Ended September 30, |
|------------------------------------------------|-----------------------------------|----------------------------------|
|                                                | 2023          | 2022          | 2023          | 2022          |
| Automotive sales                               | $18,582      | $17,785      | $57,879      | $46,969      |
| Automotive regulatory credits                   | $554         | $286         | $1,357       | $1,309       |
| Energy generation and storage sales            | $1,416       | $966         | $4,188       | $2,186       |
| Services and other                             | $2,166       | $1,645       | $6,153       | $4,390       |
| **Total revenues from sales and services**    | **$22,718**  | **$20,682**  | **$69,577**  | **$54,854**  |
| Automotive leasing                              | $489         | $621         | $1,620       | $1,877       |
| Energy generation and storage leasing           | $143         | $151         | $409         | $413         |
| **Total revenues**                             | **$23,350**  | **$21,454**  | **$71,606**  | **$57,144**  |

"""

# summarize_table(test_md)
# Define the path to the folder with markdown files
# path = "/Users/oga/Desktop/gwu_stuff/Masters Stuff/LLM Research/md_files/Tesla/2024/10Q_10K/10Q-Q2-2024.pdf.md"
# with open(path, 'r', encoding='utf-8') as file:
#     content = file.read()
# markdown_text = content
# chunks = chunk_markdown(markdown_text, max_chunk_length=512, verbose=True)

# print("done")
# # for i, chunk in enumerate(chunks):
# #     print(f"Chunk {i + 1}:\n{chunk}\n")
# # with open("output.md", "w") as f:
# #     for chunk in chunks:
# #         f.write(str(chunk) + "\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n")  # Separator between chunks


# generate_embedding(text=test_md)

authors = {
    "test": []
}
author = authors.get("test", ["Unknown"])


if(not author):
    print("no author")
else:
    print(author)