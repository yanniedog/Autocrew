# filename: logging_config.py

import logging

def setup_logging(log_file='autocrew.log'):
    logger = logging.getLogger()

    # Check if handlers already exist to prevent duplicates
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)d] %(message)s')
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        
def flush_log_handlers():
    for handler in logging.getLogger().handlers:
        handler.flush()
