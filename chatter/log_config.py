import logging
from logging.config import dictConfig
from rich.logging import RichHandler

envirment = "dev"

def logger_config():
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {},
            "formatters": {
                "standard": {
                    "class": "logging.Formatter",
                    "datefmt": "%Y-%m-%dT%H:%M:%S",
                    "format": " %(asctime)8s  %(levelname)s [%(pathname)s] line (%(lineno)d)| %(message)s "
                },
                "user_interaction": {
                    "class": "logging.Formatter",
                    "format": "%(asctime)s | %(message)s",  # Custom format for user interaction logs
                    "datefmt": "%Y-%m-%dT%H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "rich.logging.RichHandler",
                    "formatter": "standard",
                },
                "app_file": {
                    "class": "logging.FileHandler",
                    "filename": "app.log",  # Main application logs
                    "formatter": "standard",
                },
                "user_file": {
                    "class": "logging.FileHandler",
                    "filename": "user_interactions.log",  # User interaction logs
                    "formatter": "user_interaction",
                },
            },
            "loggers": {
                "app_logger": {
                    "level": "DEBUG" if envirment == "dev" else "INFO",
                    "handlers": ["app_file", "console"],
                },
                "user_logger": {
                    "level": "INFO",
                    "handlers": ["user_file"],
                },
            }
        }
    )

logger_config()

# Use the loggers in your code
app_logger = logging.getLogger("app_logger")
user_logger = logging.getLogger("user_logger")

# Example of logging user interaction
def log_user_interaction(user_id, username,interaction_type):
    message = f"{user_id} | {username} |{interaction_type}"
    user_logger.info(message)


