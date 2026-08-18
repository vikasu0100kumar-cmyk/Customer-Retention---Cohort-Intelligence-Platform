"""
Retention metrics, statistical analysis, and business KPIs.

Implements three distinct retention definitions, confidence intervals,
outlier detection, and Customer Lifetime Value estimation.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("cohort_analysis")


# ---------------------------------------------------------------------------
# Retention Metrics
# ---------------------------------------------------------------------------

def activity_rate(
    activity_matrix: pd.DataFrame,
    cohort_sizes: pd.Series,
) -> pd.DataFrame:
    """
    Activity rate: % of cohort active in each period.

    This is what the original project incorrectly called "retention rate."
    It measures: unique_customers_in_period / total_cohort_size.

    Note: This does NOT distinguish between continuously active and
    returning customers.

    Args:
        activity_matrix: Pivot of unique customers per cohort × period.
        cohort_sizes: Series of total unique customers per cohort.

    Returns:
        DataFrame with activity rates (0.0 to 1.0).
    """
    rates = activity_matrix.divide(cohort_sizes, axis=0)
    logger.info("Computed activity rates")
    return rates


def classic_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic (period-over-period) retention rate.

    For each cohort and period N (where N >= 1):
        retained_N = customers active in BOTH period N and period N-1
        retention_N = retained_N / active_in_period_(N-1)

    Period 0 is always 100% (by definition).

    Args:
        df: DataFrame with customer_id, cohort_month, cohort_index.

    Returns:
        Pivot table of classic retention rates per cohort × period.
    """
    logger.info("Computing classic (period-over-period) retention...")

    # Get unique customer-period pairs
    active = (
        df[["customer_id", "cohort_month", "cohort_index"]]
        .drop_duplicates()
    )

    cohorts = sorted(active["cohort_month"].unique())
    results = []

    for cohort in cohorts:
        cohort_data = active[active["cohort_month"] == cohort]
        periods = sorted(cohort_data["cohort_index"].unique())

        # Period 0 retention is 100%
        row = {"cohort_month": cohort, 0: 1.0}

        for i in range(1, len(periods)):
            period = periods[i]
            prev_period = periods[i - 1]

            active_prev = set(
                cohort_data[cohort_data["cohort_index"] == prev_period]["customer_id"]
            )
            active_curr = set(
                cohort_data[cohort_data["cohort_index"] == period]["customer_id"]
            )

            if len(active_prev) > 0:
                retained = len(active_curr & active_prev)
                row[period] = retained / len(active_prev)
            else:
                row[period] = np.nan

        results.append(row)

    result_df = pd.DataFrame(results).set_index("cohort_month").sort_index()
    result_df.columns.name = "cohort_index"
    logger.info(f"Classic retention matrix: {result_df.shape}")
    return result_df


def rolling_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling (unbounded) retention rate.

    For each cohort and period N:
        rolling_retained_N = customers active in period N OR ANY LATER period
        rolling_retention_N = rolling_retained_N / cohort_size

    This is useful for measuring long-term customer value — a customer
    who comes back months later is still counted as "retained."

    Args:
        df: DataFrame with customer_id, cohort_month, cohort_index.

    Returns:
        Pivot table of rolling retention rates per cohort × period.
    """
    logger.info("Computing rolling (unbounded) retention...")

    active = (
        df[["customer_id", "cohort_month", "cohort_index"]]
        .drop_duplicates()
    )

    cohorts = sorted(active["cohort_month"].unique())
    results = []

    for cohort in cohorts:
        cohort_data = active[active["cohort_month"] == cohort]
        cohort_size = cohort_data["customer_id"].nunique()
        max_period = cohort_data["cohort_index"].max()

        row = {"cohort_month": cohort}
        for period in range(int(max_period) + 1):
            # Customers active in period OR any later period
            active_on_or_after = cohort_data[
                cohort_data["cohort_index"] >= period
            ]["customer_id"].nunique()
            row[period] = active_on_or_after / cohort_size if cohort_size > 0 else np.nan

        results.append(row)

    result_df = pd.DataFrame(results).set_index("cohort_month").sort_index()
    result_df.columns.name = "cohort_index"
    logger.info(f"Rolling retention matrix: {result_df.shape}")
    return result_df


# ---------------------------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------------------------

def wilson_confidence_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    More accurate than the normal approximation, especially for small
    sample sizes or proportions near 0 or 1.

    Args:
        successes: Number of successes (e.g., retained customers).
        trials: Number of trials (e.g., cohort size).
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if trials == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / trials

    denominator = 1 + z ** 2 / trials
    center = (p_hat + z ** 2 / (2 * trials)) / denominator
    margin = (z / denominator) * np.sqrt(
        p_hat * (1 - p_hat) / trials + z ** 2 / (4 * trials ** 2)
    )

    return (max(0.0, center - margin), min(1.0, center + margin))


def retention_with_confidence(
    activity_matrix: pd.DataFrame,
    cohort_sizes: pd.Series,
    confidence: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute activity rates with Wilson confidence intervals.

    Args:
        activity_matrix: Unique customers per cohort × period.
        cohort_sizes: Total customers per cohort.
        confidence: Confidence level.

    Returns:
        Tuple of (rates, lower_bounds, upper_bounds) DataFrames.
    """
    logger.info(f"Computing retention with {confidence*100:.0f}% confidence intervals...")

    rates = activity_matrix.divide(cohort_sizes, axis=0)
    lower = rates.copy()
    upper = rates.copy()

    for cohort in activity_matrix.index:
        n = cohort_sizes.get(cohort, 0)
        if n == 0:
            continue
        for period in activity_matrix.columns:
            k = activity_matrix.loc[cohort, period]
            if pd.isna(k):
                lower.loc[cohort, period] = np.nan
                upper.loc[cohort, period] = np.nan
            else:
                lo, hi = wilson_confidence_interval(int(k), int(n), confidence)
                lower.loc[cohort, period] = lo
                upper.loc[cohort, period] = hi

    logger.info("Confidence intervals computed")
    return rates, lower, upper


def detect_outliers_iqr(
    series: pd.Series, factor: float = 1.5
) -> pd.Series:
    """
    Detect outliers using the IQR method.

    Args:
        series: Numeric series to check.
        factor: IQR multiplier (default 1.5 for standard, 3.0 for extreme).

    Returns:
        Boolean series where True indicates an outlier.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    outliers = (series < lower) | (series > upper)
    n_outliers = outliers.sum()
    if n_outliers > 0:
        logger.info(
            f"IQR outlier detection: {n_outliers:,} outliers "
            f"(bounds: [{lower:.2f}, {upper:.2f}])"
        )
    return outliers


# ---------------------------------------------------------------------------
# Business Metrics
# ---------------------------------------------------------------------------

def compute_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Customer Lifetime Value metrics per cohort.

    CLV components:
        - total_revenue: Sum of all revenue from the cohort
        - avg_revenue_per_customer: Total revenue / unique customers
        - avg_orders_per_customer: Total orders / unique customers
        - avg_order_value: Total revenue / total orders
        - customer_count: Unique customers in the cohort

    Args:
        df: DataFrame with cohort_month, customer_id, revenue, order_id.

    Returns:
        DataFrame indexed by cohort_month with CLV metrics.
    """
    logger.info("Computing Customer Lifetime Value per cohort...")

    clv = df.groupby("cohort_month").agg(
        total_revenue=("revenue", "sum"),
        total_orders=("order_id", "nunique"),
        customer_count=("customer_id", "nunique"),
        avg_order_value=("revenue", "mean"),
        total_quantity=("quantity", "sum"),
    ).sort_index()

    clv["avg_revenue_per_customer"] = clv["total_revenue"] / clv["customer_count"]
    clv["avg_orders_per_customer"] = clv["total_orders"] / clv["customer_count"]

    logger.info(f"CLV computed for {len(clv)} cohorts")
    return clv


def revenue_per_cohort_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total revenue by cohort and order month.

    Args:
        df: DataFrame with cohort_month, cohort_index, revenue.

    Returns:
        Pivot table: cohort_month × cohort_index → total revenue.
    """
    matrix = pd.pivot_table(
        df,
        index="cohort_month",
        columns="cohort_index",
        values="revenue",
        aggfunc="sum",
    ).sort_index()
    logger.info(f"Revenue-over-time matrix: {matrix.shape}")
    return matrix


def retention_revenue_correlation(
    activity_rates: pd.DataFrame,
    revenue_matrix: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute correlation between retention rates and revenue.

    Flattens both matrices and computes Pearson and Spearman correlations.

    Args:
        activity_rates: Activity rate matrix.
        revenue_matrix: Revenue matrix (same dimensions).

    Returns:
        Dict with pearson_r, pearson_p, spearman_r, spearman_p.
    """
    # Align shapes and flatten
    common_idx = activity_rates.index.intersection(revenue_matrix.index)
    common_cols = activity_rates.columns.intersection(revenue_matrix.columns)

    rates_flat = activity_rates.loc[common_idx, common_cols].values.flatten()
    rev_flat = revenue_matrix.loc[common_idx, common_cols].values.flatten()

    # Remove NaN pairs
    mask = ~(np.isnan(rates_flat) | np.isnan(rev_flat))
    rates_clean = rates_flat[mask]
    rev_clean = rev_flat[mask]

    if len(rates_clean) < 3:
        logger.warning("Insufficient data for correlation analysis")
        return {"pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan}

    pearson_r, pearson_p = stats.pearsonr(rates_clean, rev_clean)
    spearman_r, spearman_p = stats.spearmanr(rates_clean, rev_clean)

    logger.info(
        f"Retention-Revenue correlation: "
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), "
        f"Spearman ρ={spearman_r:.3f} (p={spearman_p:.4f})"
    )

    return {
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
    }
