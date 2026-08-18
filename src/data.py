"""
Data pipeline: loading, validation, cleaning, and feature engineering.

This module handles all data transformations from raw CSV to analysis-ready
DataFrame. It strips PII, filters invalid orders, validates data integrity,
and creates the features needed for cohort analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .utils import ANALYSIS_COLUMNS, PII_COLUMNS, VALID_ORDER_STATUSES

logger = logging.getLogger("cohort_analysis")


def load_data(filepath: str = "sales.csv") -> pd.DataFrame:
    """
    Load raw transaction data from CSV.

    Args:
        filepath: Path to the sales CSV file.

    Returns:
        Raw DataFrame with all columns.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    return df


def validate_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Run comprehensive data quality checks and log a report.

    Does NOT modify the DataFrame — this is a read-only audit.

    Args:
        df: Raw DataFrame to validate.

    Returns:
        Dictionary with quality metrics for downstream use.
    """
    report = {}

    # --- Shape ---
    report["total_rows"] = len(df)
    report["total_columns"] = len(df.columns)
    logger.info(f"Data shape: {len(df):,} rows × {len(df.columns)} columns")

    # --- Null check ---
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    report["null_columns"] = len(null_cols)
    if len(null_cols) > 0:
        logger.warning(f"Columns with nulls: {dict(null_cols)}")
    else:
        logger.info("No null values detected")

    # --- Duplicate check ---
    exact_dupes = df.duplicated().sum()
    report["exact_duplicates"] = exact_dupes
    logger.info(f"Exact duplicate rows: {exact_dupes:,}")

    # --- Order status distribution ---
    status_counts = df["status"].value_counts()
    report["status_distribution"] = status_counts.to_dict()
    total = len(df)
    valid_count = df[df["status"].isin(VALID_ORDER_STATUSES)].shape[0]
    report["valid_orders"] = valid_count
    report["invalid_orders"] = total - valid_count
    logger.info(
        f"Valid orders (complete/received): {valid_count:,} "
        f"({valid_count / total * 100:.1f}%)"
    )
    logger.info(
        f"Invalid orders (canceled/refunded/other): {total - valid_count:,} "
        f"({(total - valid_count) / total * 100:.1f}%)"
    )

    # --- Price/quantity sanity ---
    negative_prices = (df["price"] < 0).sum()
    zero_prices = (df["price"] == 0).sum()
    negative_totals = (df["total"] < 0).sum()
    report["negative_prices"] = negative_prices
    report["zero_prices"] = zero_prices
    report["negative_totals"] = negative_totals

    if negative_prices > 0:
        logger.warning(f"Rows with negative prices: {negative_prices:,}")
    if negative_totals > 0:
        logger.warning(f"Rows with negative totals: {negative_totals:,}")

    # --- Date range ---
    order_dates = pd.to_datetime(df["order_date"])
    report["order_date_min"] = str(order_dates.min())
    report["order_date_max"] = str(order_dates.max())
    logger.info(f"Order date range: {order_dates.min()} to {order_dates.max()}")

    # --- Customer count ---
    report["unique_customers"] = df["cust_id"].nunique()
    report["unique_orders"] = df["order_id"].nunique()
    logger.info(f"Unique customers: {report['unique_customers']:,}")
    logger.info(f"Unique orders: {report['unique_orders']:,}")

    # --- Formula verification ---
    # In this dataset: value = (qty_ordered - 1) * price
    #                  total = value - discount_amount
    mask_price_positive = df["price"] > 0
    df_check = df[mask_price_positive]
    value_matches = (
        np.abs(df_check["value"] - (df_check["qty_ordered"] - 1) * df_check["price"])
        < 0.01
    ).sum()
    total_matches = (
        np.abs(df_check["total"] - (df_check["value"] - df_check["discount_amount"]))
        < 0.01
    ).sum()
    report["value_formula_match_pct"] = value_matches / len(df_check) * 100
    report["total_formula_match_pct"] = total_matches / len(df_check) * 100
    logger.info(
        f"Formula verification: value=(qty-1)*price matches "
        f"{report['value_formula_match_pct']:.1f}% of rows"
    )
    logger.info(
        f"Formula verification: total=value-discount matches "
        f"{report['total_formula_match_pct']:.1f}% of rows"
    )

    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data for analysis.

    Steps:
        1. Strip all PII columns
        2. Filter to valid order statuses only (complete, received)
        3. Parse date columns
        4. Derive true quantity from value/price (source data has off-by-one)
        5. Remove zero-revenue transactions
        6. Rename columns for clarity

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    logger.info("--- Cleaning pipeline started ---")
    initial_rows = len(df)

    # Step 1: Strip PII
    pii_present = [col for col in PII_COLUMNS if col in df.columns]
    df = df.drop(columns=pii_present)
    logger.info(f"Stripped {len(pii_present)} PII columns: {pii_present}")

    # Step 2: Filter to valid orders only
    df = df[df["status"].isin(VALID_ORDER_STATUSES)].copy()
    logger.info(
        f"Filtered to valid orders: {len(df):,} rows "
        f"(removed {initial_rows - len(df):,} canceled/refunded)"
    )

    # Step 3: Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"], format="%Y-%m-%d")
    df["customer_since"] = pd.to_datetime(df["Customer Since"], format="%m/%d/%Y")
    df = df.drop(columns=["Customer Since"])

    # Step 4: Derive true quantity
    # Source data: value = (qty_ordered - 1) * price, so true_qty = qty_ordered - 1
    # For rows where price == 0, we keep qty_ordered as-is (free items)
    safe_qty = np.where(
        df["price"] > 0,
        (df["value"] / df["price"]).round(),
        df["qty_ordered"],
    )
    df["quantity"] = np.nan_to_num(safe_qty, nan=0.0, posinf=0.0, neginf=0.0).astype(int)
    logger.info(
        f"Derived true quantity from value/price "
        f"(source qty_ordered has systematic +1 offset)"
    )

    # Step 5: Remove zero-revenue and zero-quantity transactions
    before = len(df)
    df = df[(df["total"] > 0) & (df["quantity"] > 0)]
    logger.info(
        f"Removed {before - len(df):,} zero-revenue/zero-quantity rows → "
        f"{len(df):,} rows remain"
    )

    # Step 6: Assert data integrity
    assert (df["quantity"] > 0).all(), "Quantity must be positive after cleaning"
    assert (df["price"] >= 0).all(), "Price must be non-negative"
    assert (df["total"] > 0).all(), "Total must be positive after cleaning"
    assert df["order_date"].notna().all(), "Order dates must not be null"
    assert df["cust_id"].notna().all(), "Customer IDs must not be null"

    # Step 7: Select and rename final columns
    df = df[
        [
            "order_id", "order_date", "cust_id", "customer_since",
            "quantity", "price", "total", "discount_amount",
            "category", "Region", "age", "Gender",
        ]
    ].copy()
    df = df.rename(columns={
        "cust_id": "customer_id",
        "Region": "region",
        "Gender": "gender",
    })

    logger.info(f"--- Cleaning complete: {len(df):,} rows × {len(df.columns)} cols ---")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add analysis-ready features.

    Creates:
        - order_month: Period representation of order date (YYYY-MM)
        - revenue: Alias for total (explicit naming)

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with added feature columns.
    """
    logger.info("Adding engineered features...")

    df = df.copy()

    # Order month as period for grouping
    df["order_month"] = df["order_date"].dt.to_period("M")

    # Revenue column (explicit alias for total)
    df["revenue"] = df["total"]

    logger.info(
        f"Features added: order_month, revenue | "
        f"Final shape: {df.shape[0]:,} × {df.shape[1]}"
    )
    return df


def run_pipeline(filepath: str = "sales.csv") -> Tuple[pd.DataFrame, Dict]:
    """
    Execute the full data pipeline: load → validate → clean → engineer.

    This is the primary entry point for data preparation.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Tuple of (analysis-ready DataFrame, validation report dict).
    """
    df_raw = load_data(filepath)
    report = validate_data(df_raw)
    df_clean = clean_data(df_raw)
    df_final = feature_engineering(df_clean)
    return df_final, report
