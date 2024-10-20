from transformers import AutoTokenizer
import ollama
import re
from funcs import write_debug_log
import math

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

def check_nearest_punctuation(strlist, front = True):
    """
    For a list of strings, return the number of strings before the first occurance of a string containing '.', '!', or '?'

    Args:
        strlist list(str): list of words to check
        front bool: flag for whether to check iterating from the start of the end 

    Returns:
        ret int: number of words before first occurance. Will return None if nothing is found
    """
    sentence_endings = {'.', '!', '?'}
    if front:
        for i, word in enumerate(strlist):
            if any(char in sentence_endings for char in word):
                return i+1
    else:
        for i, word in enumerate(reversed(strlist)):
            if any(char in sentence_endings for char in word):
                return i        

def split_section(text, num_chunks):
    """
    For a section taken from markdown, split it into num_chunks smaller chunks (will not split apart sentences)

    Args:
        text str: string representation of the section of md
        num_chunks int: integer representing how many parts to split the section into

    Returns:
        ret list(str): list of strings
    """
    # split into individual words
    words = text.split(' ')
    
    avg_size = len(words) // num_chunks
    remainder = len(words) % num_chunks  # Calculate the extra elements to distribute

    sublists = []
    start = 0

    for i in range(num_chunks):
        # Determine the size of the current sublist
        sublist_size = avg_size + (1 if i < remainder else 0)
        sublists.append(words[start:start + sublist_size])
        start += sublist_size
    
    # Now sublists contains an even split of words. 
    # try and rejoin sentences together 
    for i in range(num_chunks-1):
        ending_words_num = check_nearest_punctuation(sublists[i], front = False)
        starting_words_num = check_nearest_punctuation(sublists[i+1], front = True)
        
        if(ending_words_num and starting_words_num and ending_words_num < starting_words_num):
            # move the end of list i to start of list i+1
            elements_to_move = sublists[i][-ending_words_num:]
            sublists[i] = sublists[i][:-ending_words_num]
            sublists[i+1] = elements_to_move + sublists[i+1]
        elif(ending_words_num and starting_words_num and ending_words_num > starting_words_num):
            elements_to_move = sublists[i+1][:starting_words_num]
            sublists[i+1] = sublists[i+1][starting_words_num:]
            sublists[i] = sublists[i] + elements_to_move
    
    ret = []
    for l in sublists:
        ret.append(" ".join(l))

    return ret

def verify_chunks(chunks_list, max_chunk_length):
    """
    Take a list of chunks returned by split_section() and verify that they are valid sizes. If they are not, 
    perform manipulation to make the chunks fit the desired size. 

    Args:
        chunks_list list(str): list of chunks in string format
        max_chunk_length int: max length of chunk that can be handled in terms of number of tokens
        overlap float: float indicating how much of each chunk to overlap

    Returns:
        chunks_list list(str): list of chunks in string format that have been verified and corrected
        bool: T/F statement that tells whether the returned chunks_list contains all valid chunks or not
    """
    
    for i in range(len(chunks_list)):
        check_token_len = num_tokens(chunks_list[i], tokenizer)
        # If under min token length, grab sentences from neighboring chunks to make chunk larger
        if(check_token_len < MIN_CHUNK_LEN):
            # If first chunk, grab sentence from next chunk
            if(i == 0):
                sentences = re.split(r'(?<=[.!?])\s+', chunks_list[i+1].strip())
                # keep grabbing sentences until chunk is long enough
                for sentence in sentences:
                    chunks_list[i] += " " + sentence
                    if (num_tokens(chunks_list[i], tokenizer) > MIN_CHUNK_LEN):
                        break
            # If Last chunk, grab from previous chunk
            elif(i >= len(chunks_list)-1):
                sentences = re.split(r'(?<=[.!?])\s+', chunks_list[i-1].strip())
                # keep grabbing sentences until chunk is long enough
                for sentence in reversed(sentences):
                    chunks_list[i] = sentence + " " + chunks_list[i]
                    if (num_tokens(chunks_list[i], tokenizer) > MIN_CHUNK_LEN):
                        break
            # If chunk in the middle, take shortest sentence from either neighboring chunk
            else:  
                before_sentences = re.split(r'(?<=[.!?])\s+', chunks_list[i-1].strip())
                after_sentences = re.split(r'(?<=[.!?])\s+', chunks_list[i+1].strip())
                while(len(before_sentences) > 0 and len(after_sentences) > 0):
                    if(len(before_sentences[-1]) > len(after_sentences[0])):
                        chunks_list[i] += " " + after_sentences[0]
                        after_sentences.pop(0)
                        if (num_tokens(chunks_list[i], tokenizer) > MIN_CHUNK_LEN):
                            break
                    else:
                        chunks_list[i] = before_sentences[-1] + " " + chunks_list[i]
                        before_sentences.pop(-1)
                        if (num_tokens(chunks_list[i], tokenizer) > MIN_CHUNK_LEN):
                            break
        # If chunk is too large, set flag so that c+1 chunking can happen
        if(check_token_len > max_chunk_length ):
            return chunks_list, False
    return chunks_list, True

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
    token_len = num_tokens(text, tokenizer)
    if (token_len > max_chunk_length):
        
        correct_chunking = True

        # calculate how many chunks are needed
        c = math.ceil(token_len / max_chunk_length)
        # try to chunk for c chunks
        ret_chunks = split_section(text, c)
        
        ret_chunks, correct_chunking = verify_chunks(ret_chunks, max_chunk_length)
        
        if(not correct_chunking):
            # increment chunks by 1
            c = c + 1
            # try to chunk for c chunks
            ret_chunks = split_section(text, c)
            
            ret_chunks, correct_chunking = verify_chunks(ret_chunks, max_chunk_length)
            if(not correct_chunking):
                print("Need to implement a third case to split sentences so chunks dont overflow")
                raise
        
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
        # print("meow2")
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
            # TODO: Check whether section buffer contains any lines other than header lines to prevent header->table->header edge case
            all_section_header = True
            for added_line in section_buffer:
                if (not is_section_header(added_line)):
                    all_section_header = False
                    break
            if (all_section_header):
                section_buffer = []
                add_line(section_buffer, line)
                continue
                
            # Check section length to prevent small chunk sizes
            if(num_tokens(" ".join(section_buffer), tokenizer) < MIN_CHUNK_LEN):
                # If so, continue parsing to add the next section
                add_line(section_buffer, line)
                continue
            # Check section length to prevent large chunk sizes
            elif(num_tokens(" ".join(section_buffer), tokenizer) > max_chunk_length):
                # if so break into smaller chunks
                # TODO: Rewrite chunk_section
                chunks.extend(chunk_section(section_buffer, max_chunk_length))
                section_buffer = []
                add_line(section_buffer, line)
            # If section length is in between min and max length, add to chunks and continue
            else: 
                chunks.extend(chunk_section(section_buffer, max_chunk_length))
                section_buffer = []
                add_line(section_buffer, line)
            
         # if not check if line is a table
        elif is_table(line):
            # TODO: Rewrite hangle_table to not use while loop
            table_chunk, line = handle_table(lines, line)
            chunks.append((table_chunk, summarize_table(table_chunk, section_buffer)))
        # Otherwise, line must be a normal text line, so add it to the section bufer
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
path = "/Users/oga/Desktop/gwu_stuff/Masters Stuff/LLM Research/md_files/Tesla/2024/10Q_10K/10Q-Q2-2024.pdf.md"
with open(path, 'r', encoding='utf-8') as file:
    content = file.read()
markdown_text = content
chunks = chunk_markdown(markdown_text, max_chunk_length=MAX_CHUNK_LEN, verbose=True)
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}:\n{chunk}\n")
with open("output.md", "w") as f:
    for chunk in chunks:
        f.write(str(chunk) + "\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n")  # Separator between chunks
