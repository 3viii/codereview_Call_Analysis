import logging
import sys

def setup_logging(level=logging.INFO):
    """
    Configure logging for the application.
    """
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_logger(name: str):
    return logging.getLogger(name)
