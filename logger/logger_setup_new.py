import logging.config

# Loading the logging configuration from the config file
logging.config.fileConfig('../chatbot-dev/logger/logging_config.ini')
#logging.config.fileConfig("logger\logging_config.ini")

# Creating a logger using the same name as in the config file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)