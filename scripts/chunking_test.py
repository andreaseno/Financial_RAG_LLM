from chunking import chunk_markdown

file_path = "/Users/oga/Desktop/gwu_stuff/Masters Stuff/LLM Research/md_files/Nvidia/2022/10Q_10K/10Q-Q1-2022.pdf.md"

with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
chunks = chunk_markdown(content,512, verbose=False)
with open("output.md", "w") as f:
    for chunk in chunks:
        f.write(str(chunk) + "\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n")  # Separator between chunks
