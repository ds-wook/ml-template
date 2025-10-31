import logging


def setup_logger(log_file: str):
    # Create a logger object
    logger = logging.getLogger("toss")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to prevent duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler(log_file, mode="w")  # Open file in write mode to overwrite on each run

    # Set a logging format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Create a console handler to log messages to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
