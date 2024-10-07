import os
from dotenv import load_dotenv
import shutil
# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
# import llamaparse  # Assuming llamaparse has a Python API. Replace with actual import if needed.

# Load environment variables from .env file
load_dotenv()

# Define the source and destination directories
source_dir = "../pdf_files"
dest_dir = "../md_files"

parsingInstruction = """The provided document is a finincial document. It contains both text and tables. 
When parsing, extract section headers as H1 or H2 (# or ##) markdown headings, and extract subsection headers as H3 or H4 depending on their depth.
Exclude paragraphs containing information about "Safe Harbor" or "Forward-Looking Statements." Also ignore document metadata tables (e.g., “Document Prepared By”).
Convert tables into markdown table format. Ensure headers are included and numeric data is aligned properly. For instance:
| Revenue | Expenses | Profit |
|---------|----------|--------|
Summarize any legal disclaimers or complex legal clauses in a single sentence, then move on.
Convert any html characters such as "&nbsp;" to their markdown equivalent. 
"""

# set up parser
parser = LlamaParse(
    result_type="markdown",  # "markdown" and "text" are available
    parsing_instruction=parsingInstruction
)

# Function to parse a PDF file into markdown using llamaparse
def parse_pdf_to_md(pdf_path, md_output_path):
    # use SimpleDirectoryReader to parse our file
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[pdf_path], file_extractor=file_extractor).load_data()
    print()
    # Write markdown content to the specified output path
    full_text = ""
    for i, doc in enumerate(documents):
        full_text += doc.text
        full_text += "\n>>>>>>>>>\n"
    with open(md_output_path, "w") as md_file:
        md_file.write(full_text)

# Function to walk through source directory, create corresponding directories in dest_dir, and parse PDF files
def process_directory(source, destination):
    for root, dirs, files in os.walk(source):
        # Create corresponding directory structure in dest_dir
        relative_path = os.path.relpath(root, source)
        new_dest_dir = os.path.join(destination, relative_path)
        os.makedirs(new_dest_dir, exist_ok=True)

        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                md_filename = os.path.splitext(file)[0] + ".pdf.md"
                md_output_path = os.path.join(new_dest_dir, md_filename)

                # Check if the markdown file already exists
                if not os.path.exists(md_output_path):
                    # Parse the PDF and save as markdown
                    print(f"Parsing: {pdf_path} -> {md_output_path}")
                    parse_pdf_to_md(pdf_path, md_output_path)
                else:
                    print(f"Skipping: {md_output_path} already exists")

# Ensure the destination directory exists
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Start processing the directory
process_directory(source_dir, dest_dir)
print("----------------\nFINISHED PARSING\n----------------")
