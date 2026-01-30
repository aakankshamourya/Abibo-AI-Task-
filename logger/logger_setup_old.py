import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler for logging
log_filename = "../chatbot-dev/logger/ask-chatbot-dev-nexus-aspire-ds.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)

# Create a formatter with a timestamp and add it to the file handler
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
