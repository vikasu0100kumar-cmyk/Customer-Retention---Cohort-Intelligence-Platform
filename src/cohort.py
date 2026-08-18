"""
Cohort assignment and matrix construction.

This module handles the core cohort logic:
    - Assigning each customer to their first-purchase cohort
    - Computing the cohort index (months since first purchase)
    - Building pivot matrices for any metric
"""

import logging

import pandas as pd

logger = logging.getLogger("cohort_analysis")


def assign_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each customer to a cohort based on their FIRST purchase month.

    This is the corrected approach — the original used account creation date
    ('Customer Since'), which spans 1978–2017 and doesn't represent actual
    purchasing behavior.

    Creates:
        - cohort_month: The month of the customer's first purchase (Period)
        - cohort_index: Number of months since the customer's first purchase

    Args:
        df: Cleaned DataFrame with order_date, customer_id, order_month.

    Returns:
        DataFrame with cohort_month and cohort_index columns added.
    """
    logger.info("Assigning cohorts based on first purchase date...")
    df = df.copy()

    # Derive cohort from first purchase (NOT account creation)
    first_purchase = (
        df.groupby("customer_id")["order_date"]
        .min()
        .reset_index()
        .rename(columns={"order_date": "first_purchase_date"})
    )
    first_purchase["cohort_month"] = first_purchase["first_purchase_date"].dt.to_period("M")

    df = df.merge(first_purchase, on="customer_id", how="left")

    # Cohort index: how many months after first purchase is this order?
    df["cohort_index"] = (
        df["order_month"].dt.year * 12 + df["order_month"].dt.month
    ) - (
        df["cohort_month"].dt.year * 12 + df["cohort_month"].dt.month
    )

    n_cohorts = df["cohort_month"].nunique()
    max_index = df["cohort_index"].max()
    logger.info(
        f"Assigned {n_cohorts} cohorts | "
        f"Cohort index range: 0–{max_index} months"
    )

    return df


def get_cohort_sizes(df: pd.DataFrame) -> pd.Series:
    """
    Count unique customers per cohort.

    Args:
        df: DataFrame with cohort_month and customer_id.

    Returns:
        Series indexed by cohort_month with customer counts.
    """
    sizes = (
        df.groupby("cohort_month")["customer_id"]
        .nunique()
        .sort_index()
    )
    logger.info(f"Cohort sizes computed for {len(sizes)} cohorts")
    return sizes


def build_cohort_matrix(
    df: pd.DataFrame,
    value_col: str,
    agg_func: str = "nunique",
) -> pd.DataFrame:
    """
    Build a generic cohort pivot table.

    Args:
        df: DataFrame with cohort_month and cohort_index columns.
        value_col: Column to aggregate.
        agg_func: Aggregation function ('nunique', 'sum', 'mean', 'median').

    Returns:
        Pivot table with cohort_month as rows, cohort_index as columns.
    """
    matrix = pd.pivot_table(
        df,
        index="cohort_month",
        columns="cohort_index",
        values=value_col,
        aggfunc=agg_func,
    )
    matrix = matrix.sort_index()
    logger.info(
        f"Built cohort matrix: {value_col} ({agg_func}) | "
        f"Shape: {matrix.shape}"
    )
    return matrix


def build_customer_activity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a matrix showing unique active customers per cohort per month.

    This is the foundation for all retention calculations.

    Args:
        df: DataFrame with cohort assignments.

    Returns:
        Pivot table: cohort_month × cohort_index → unique customer count.
    """
    return build_cohort_matrix(df, value_col="customer_id", agg_func="nunique")


def get_customer_cohort_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a customer-level activity table showing which cohort indices
    each customer was active in.

    Args:
        df: DataFrame with cohort assignments.

    Returns:
        DataFrame with columns: customer_id, cohort_month, cohort_index
        (one row per unique customer-month combination).
    """
    activity = (
        df.groupby(["customer_id", "cohort_month", "cohort_index"])
        .size()
        .reset_index(name="transaction_count")
    )
    return activity
