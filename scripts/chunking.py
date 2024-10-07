from transformers import AutoTokenizer
import ollama
import re
from funcs import write_debug_log

# Initialize the Ollama client
client = ollama.Client()

# Load the tokenizer and the base model (without classification head)
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

# Both in tokens
MAX_CHUNK_LEN = 512
MIN_CHUNK_LEN = 300
DEBUG = True

def summarize_table(table, section_buffer):
    """
    Uses Ollama3.1 to create a summary for a table 

    Args:
        table str: markdown representation of the table
        section_buffer list(str): list of previous lines from section table was taken from

    Returns:
        strings list(str): list of strings
    """
    
    message = f"""The given table context and contains a chunk of information from a 10Q/10K financial document. Specifically, it contains a table in 
    markdown format with preceding context. Give a 1 to 3 sentence summary of the document based on the column names, row names, and preceding context in 200 or less words.
    Do not include any specific information about the figures contained in the table.
    Your message should only contain the summary, so leave out any wording like "Here is a summary of the table in 300 words or less:"
    
    
    context:
    {" ".join(section_buffer)}
    
    text:
    {table}
    
    """
    response = ollama.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': message,},])
    return response["message"]["content"]

def process_strings(strings):
    """
    For a list of strings, make sure each string contains a full sentence 
    (move split up sentences to the previous element to prevent splitting sentences across chunks)

    Args:
        strings list(str): list of strings

    Returns:
        strings list(str): list of strings
    """
    i = 0
    while i < len(strings):
        # Remove leading and trailing whitespace from the current string
        current_string = strings[i].strip()
        
        # Check if the current string ends with sentence-ending punctuation
        if current_string and current_string[-1] in '.!?':
            # Update the string in the list
            strings[i] = current_string
            i += 1  # Move to the next string
        else:
            # Initialize index for the next string
            j = i + 1
            # Initialize a flag to check if punctuation is found
            punctuation_found = False
            
            # Loop through the following strings to find sentence-ending punctuation
            while j < len(strings):
                next_string = strings[j]
                # Find the position of the first sentence-ending punctuation
                punctuation_positions = [next_string.find(p) for p in '.!?' if next_string.find(p) != -1]
                
                if punctuation_positions:
                    # Sentence-ending punctuation found
                    first_punc_pos = min(punctuation_positions) + 1  # Include the punctuation
                    # Append the text up to the punctuation to the current string
                    strings[i] = current_string + ' ' + next_string[:first_punc_pos].strip()
                    # Update the next string by removing the moved text
                    strings[j] = next_string[first_punc_pos:].lstrip()
                    punctuation_found = True
                    break  # Exit the inner loop
                else:
                    # No punctuation found, append the entire next string to the current string
                    current_string += ' ' + next_string.strip()
                    strings[i] = current_string
                    # Remove the next string from the list
                    strings.pop(j)
                    # Do not increment j since the list has shrunk
                    continue
            if not punctuation_found:
                # No punctuation found in the following strings
                i += 1  # Move to the next string
    return strings

def split_section(text, num_chunks):
    """
    For a section taken from markdown, split it into num_chunks smaller chunks (will not split apart sentences)

    Args:
        text str: string representation of the section of md
        num_chunks int: integer representing how many parts to split the section into

    Returns:
        ret list(str): list of strings
    """
    n = len(text)
    part_size = n // num_chunks  # Size of each part
    remainder = n % num_chunks  # Handle the remainder if string length is not perfectly divisible by x
    
    parts = [text[i * part_size + min(i, remainder):(i + 1) * part_size + min(i + 1, remainder)] for i in range(num_chunks)]
    
    ret = process_strings(parts)
            
    return ret

def chunk_section(section_buffer, max_chunk_length, overlap=0.2):
    """
    Take a buffer representing a section from the md file, and split it into equal chunks

    Args:
        section_buffer list(str): buffer list where each element is a line from the md doc
        max_chunk_length int: max length of chunk that can be handled in terms of number of tokens
        overlap float: float indicating how much of each chunk to overlap

    Returns:
        ret_chunks list(str): list of strings representing chunks
    """
    # TODO: implement overlap
    text = " ".join(section_buffer)
    if (num_tokens(text, tokenizer) > max_chunk_length):
        correct_chunking = False
        num_chunks = 2
        while not correct_chunking:
            # Initialize list to hold the chunks
            ret_chunks = split_section(text, num_chunks)
            num_chunks += 1
            correct_chunking = True
            for c in ret_chunks:
                if num_tokens(c, tokenizer) > max_chunk_length:
                    correct_chunking = False
        return ret_chunks
            
    else:
        return [text]

def num_tokens(text, my_tokenizer):
    """
    Count the number of tokens in a text based on the tokenizer being used

    Args:
        text str: text to be checked
        my_tokenizer PreTrainedTokenizer: tokenizer to tokenize the text

    Returns:
        ____ int: number of tokens from text
    """
    # Tokenize the text
    tokens = my_tokenizer.encode(text)

    # Return the number of tokens
    return len(tokens)

def is_section_header(line: str) -> bool:
    """
    check if a line is a section header

    Args:
        line str: line to be checked

    Returns:
        ____ bool: whether line is header or not
    """
    # Check if the line starts with '#' followed by a space
    return line.strip().startswith('# ') and len(line.strip()) > 2

def is_table(line):
    """
    check if a line is a table

    Args:
        line str: line to be checked

    Returns:
        ____ bool: whether line is table or not
    """
    # A simple heuristic for detecting tables based on markdown table syntax
    return '|' in line or re.match(r'^[-]+', line)

def get_next_line(lines):
    """
    grab the next line in the iterable

    Args:
        lines iter: iterable

    Returns:
        line str: next line
    """
    line = next(lines, None)
    while ((not line == None) and (is_ignored_line(line))):
        line = next(lines, None)
    return line
    
def handle_table(lines, first_line):
    """
    Will make sure tables are separated out into their own chunk

    Args:
        lines iter: iterable to iterate over
        first_line str: first string from previous iteration

    Returns:
        ____ str: chunk of resulting table
    """
    # Handles reading lines from a table and returns a table chunk.
    table_buffer = [first_line]  # Start with the first line passed
    line = get_next_line(lines)
    while ((not line == None) and line and is_table(line)):
        table_buffer.append(line)
        line = get_next_line(lines)
    return '\n'.join(table_buffer), line  # Return the table chunk and the next non-table line

def is_ignored_line(line):
    """
    check if a line should be ignored

    Args:
        line str: line to be checked

    Returns:
        ____ bool: whether line is equal to >>>>>>>>>> or ```markdown or ```
    """
    ignored_patterns = [r'^>+', r'^```markdown$', r'```']
    return any(re.match(pattern, line) for pattern in ignored_patterns)

def add_line(buffer, line):
    """
    add line to buffer while checking for NoneType

    Args:
        buffer list(str): buffer to add to
        line str: line to be checked

    Returns:
        None
    """
    if (not line == None):
        buffer.append(line)

def chunk_markdown(md_text, max_chunk_length, verbose=False):
    """
    Chunk the passed markdown text

    Args:
        md_text str: entire markdown text in string format
        max_chunk_length int: max length of chunk tht can be handled in terms of number of tokens
        verbose bool: whether to show additional output

    Returns:
        chunks list(str): full list of chunks from the md file
    """
    lines = iter(md_text.split('\n'))
    chunks = []
    section_buffer = []
    
    if verbose:
        print("Chunking Started.")
    if DEBUG:
        write_debug_log("Chunking Started.")
    
    for i, line in enumerate(lines):
        if verbose:
            print(f"Chunking line {i+1}")
        if DEBUG:
            write_debug_log(f"Chunking line {i+1}")
        # Check for lines to ignore
        if is_ignored_line(line):
            continue
        if not section_buffer:
            add_line(section_buffer, line)
            continue
        # Check if new section has started
        if is_section_header(line):
            # Check section length to prevent small chunk sizes
            while ((not line == None) and (num_tokens(" ".join(section_buffer), tokenizer) < MIN_CHUNK_LEN)):
                # If so, add the next section
                next_section_buffer = []
                add_line(next_section_buffer, line)
                line = get_next_line(lines)
                while ((not line == None) and (not is_section_header(line))):
                    if is_table(line):
                        table_chunk, line = handle_table(lines, line)
                        chunks.append((table_chunk, summarize_table(table_chunk, section_buffer)))
                    else:
                        add_line(next_section_buffer, line)
                        line = get_next_line(lines)
                section_buffer.extend(next_section_buffer)
            section_chunks = chunk_section(section_buffer, max_chunk_length)
            chunks.extend(section_chunks)
            section_buffer = []
            add_line(section_buffer, line)
        # if not continue adding to section buffer
        else:
            if is_table(line):
                table_chunk, line = handle_table(lines, line)
                chunks.append((table_chunk, summarize_table(table_chunk, section_buffer)))
            else:
                # Add to the current paragraph buffer
                add_line(section_buffer, line)

    # Final flush for any remaining paragraph
    if section_buffer:
        chunks.extend(chunk_section(section_buffer, max_chunk_length))
    return chunks




# UNCOMMENT BELOW TO TEST CHUNKING ALGORITHM
# WILL OUTPUT A SINGLE CHUNKED DOCUMENT TO output.md

# Define the path to the folder with markdown files
# path = "/Users/oga/Desktop/gwu_stuff/Masters Stuff/LLM Research/md_files/Tesla/2024/10Q_10K/10Q-Q2-2024.pdf.md"
# with open(path, 'r', encoding='utf-8') as file:
#     content = file.read()
# markdown_text = content
# chunks = chunk_markdown(markdown_text, max_chunk_length=MAX_CHUNK_LEN, verbose=True)
# # for i, chunk in enumerate(chunks):
# #     print(f"Chunk {i + 1}:\n{chunk}\n")
# with open("output.md", "w") as f:
#     for chunk in chunks:
#         f.write(str(chunk) + "\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n")  # Separator between chunks
