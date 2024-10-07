import datetime

def write_debug_log(message, log_file="debug_log.txt"):
    """
    Writes a debug message to a log file with a timestamp.
    
    Args:
        message (str): The debug message to write to the log file.
        log_file (str): The file to write the debug message to (default is "debug_log.txt").
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as file:
        file.write(f"[{timestamp}] DEBUG: {message}\n")
