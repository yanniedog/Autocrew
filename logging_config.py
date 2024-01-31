# filename: logging_config.py

import logging

def setup_logging(log_file='autocrew.log'):
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the root logger level to DEBUG

    # Create handlers
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)  # Set the file handler level to DEBUG

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the console handler level to INFO

    # Create formatters and add them to handlers
    file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)d] %(message)s')
    file_handler.setFormatter(file_formatter)

    console_formatter = logging.Formatter('%(message)s')  # Only log the message
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.getLogger('httpx').setLevel(logging.WARNING)
