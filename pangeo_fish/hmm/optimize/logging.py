import logging


def create_default_formatter() -> logging.Formatter:
    """Create a default formatter of log messages.
    This function is not supposed to be directly accessed by library users.
    """
    header = "[%(levelname)1.1s %(asctime)s]"
    message = "%(message)s"
    return logging.Formatter(f"{header} {message}")


def setup_logging(logger):
    consoleHandler = logging.StreamHandler()
    formatter = create_default_formatter()
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(logging.DEBUG)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.INFO)

    return logger
