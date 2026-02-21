"""Logging setup for the application."""

import logging
import logging.config


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def configure_logging(log_level: str) -> None:
    """Configure structured logging for application services."""
    if log_level not in _VALID_LOG_LEVELS:
        raise ValueError(f"Invalid APP_LOG_LEVEL: {log_level}")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": (
                        "%(asctime)s %(levelname)s %(name)s "
                        "event=%(message)s"
                    )
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "standard",
                }
            },
            "root": {
                "level": log_level,
                "handlers": ["console"],
            },
        }
    )
