"""
Utility functions: logging configuration and shared helpers.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure and return the project-wide logger.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("cohort_analysis")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_output_dir(output_dir: str = "output") -> Path:
    """
    Create the output directory if it doesn't exist.

    Args:
        output_dir: Path to the output directory.

    Returns:
        Path object for the output directory.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


# PII columns that must be stripped from raw data before any analysis.
PII_COLUMNS = [
    "Name Prefix", "First Name", "Middle Initial", "Last Name",
    "full_name", "E Mail", "SSN", "Phone No. ", "User Name",
    "Place Name", "County", "City", "State", "Zip",
]

# Order statuses considered as completed transactions.
VALID_ORDER_STATUSES = ["complete", "received"]

# Safe columns to retain for analysis (no PII).
ANALYSIS_COLUMNS = [
    "order_id", "order_date", "status", "item_id", "sku",
    "qty_ordered", "price", "value", "discount_amount", "total",
    "category", "payment_method", "cust_id", "year", "month",
    "Gender", "age", "Region", "Discount_Percent",
]
