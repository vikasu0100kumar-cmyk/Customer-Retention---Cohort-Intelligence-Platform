#!/usr/bin/env python3
"""
Cohort Analysis: Customer Retention & Revenue Analytics
======================================================

Production-grade cohort analysis pipeline for e-commerce transaction data.

This script orchestrates the full analysis:
    1. Data loading, validation, and cleaning
    2. Cohort assignment (by first purchase month)
    3. Retention metric computation (activity rate, classic, rolling)
    4. Statistical analysis (confidence intervals, outlier detection)
    5. Business metrics (CLV, revenue trends, correlations)
    6. Visualization generation (10 publication-quality charts)

Usage:
    python main.py
    python main.py --data path/to/sales.csv --output results/
"""

import argparse
import logging
import sys
import time

from src.utils import setup_logging, ensure_output_dir
from src.data import run_pipeline
from src.cohort import (
    assign_cohorts,
    get_cohort_sizes,
    build_customer_activity_matrix,
    build_cohort_matrix,
)
from src.metrics import (
    activity_rate,
    classic_retention,
    rolling_retention,
    retention_with_confidence,
    detect_outliers_iqr,
    compute_clv,
    revenue_per_cohort_over_time,
    retention_revenue_correlation,
)
from src.visualization import generate_all_plots


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cohort Analysis: Customer Retention & Revenue Analytics",
    )
    parser.add_argument(
        "--data", type=str, default="sales.csv",
        help="Path to the sales CSV file (default: sales.csv)",
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Output directory for plots (default: output/)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main():
    """Execute the full cohort analysis pipeline."""
    args = parse_args()
    logger = setup_logging(args.log_level)
    ensure_output_dir(args.output)

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("  COHORT ANALYSIS: CUSTOMER RETENTION & REVENUE ANALYTICS")
    logger.info("=" * 70)

    # ===================================================================
    # PHASE 1: Data Pipeline
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 1: DATA PIPELINE")
    logger.info("=" * 70)

    df, validation_report = run_pipeline(args.data)

    # ===================================================================
    # PHASE 2: Cohort Assignment
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 2: COHORT ASSIGNMENT")
    logger.info("=" * 70)

    df = assign_cohorts(df)
    cohort_sizes = get_cohort_sizes(df)

    logger.info("\nCohort sizes:")
    for cohort, size in cohort_sizes.items():
        logger.info(f"  {cohort}: {size:,} customers")

    # ===================================================================
    # PHASE 3: Retention Metrics
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 3: RETENTION METRICS")
    logger.info("=" * 70)

    # Activity matrix (unique customers per cohort × period)
    activity_matrix = build_customer_activity_matrix(df)

    # Activity rate (what the original incorrectly called "retention")
    activity_rates = activity_rate(activity_matrix, cohort_sizes)
    logger.info("\nActivity rates (period 0, first 5 cohorts):")
    for cohort in list(activity_rates.index)[:5]:
        rate = activity_rates.loc[cohort, 0] if 0 in activity_rates.columns else "N/A"
        logger.info(f"  {cohort}: {rate:.1%}" if isinstance(rate, float) else f"  {cohort}: {rate}")

    # Classic retention (period-over-period)
    classic_ret = classic_retention(df)

    # Rolling retention
    rolling_ret = rolling_retention(df)

    # Activity rate with confidence intervals
    rates, ci_lower, ci_upper = retention_with_confidence(
        activity_matrix, cohort_sizes, confidence=0.95
    )

    # Log a sample CI
    if len(rates) > 0 and 1 in rates.columns:
        sample_cohort = rates.index[0]
        rate_val = rates.loc[sample_cohort, 1]
        lo = ci_lower.loc[sample_cohort, 1]
        hi = ci_upper.loc[sample_cohort, 1]
        if not any(map(lambda x: x != x, [rate_val, lo, hi])):  # NaN check
            logger.info(
                f"\nSample 95% CI for {sample_cohort}, month 1: "
                f"{rate_val:.1%} [{lo:.1%}, {hi:.1%}]"
            )

    # ===================================================================
    # PHASE 4: Business Metrics
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 4: BUSINESS METRICS")
    logger.info("=" * 70)

    # Customer Lifetime Value
    clv = compute_clv(df)
    logger.info("\nCLV Summary (Top 5 cohorts by avg revenue/customer):")
    top_clv = clv.nlargest(5, "avg_revenue_per_customer")
    for cohort, row in top_clv.iterrows():
        logger.info(
            f"  {cohort}: ${row['avg_revenue_per_customer']:,.2f}/customer, "
            f"{row['avg_orders_per_customer']:.1f} orders/customer, "
            f"{row['customer_count']:,} customers"
        )

    # Revenue over time
    revenue_matrix = revenue_per_cohort_over_time(df)

    # Quantity matrix
    quantity_matrix = build_cohort_matrix(df, value_col="quantity", agg_func="mean")

    # Retention-Revenue correlation
    correlation = retention_revenue_correlation(activity_rates, revenue_matrix)
    logger.info(
        f"\nRetention-Revenue Correlation: "
        f"Pearson r = {correlation['pearson_r']:.3f} "
        f"(p = {correlation['pearson_p']:.4f})"
    )

    # ===================================================================
    # PHASE 5: Outlier Detection
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 5: OUTLIER DETECTION")
    logger.info("=" * 70)

    # Revenue outliers per transaction
    revenue_outliers = detect_outliers_iqr(df["revenue"], factor=1.5)
    logger.info(
        f"Revenue outliers (IQR): {revenue_outliers.sum():,} of {len(df):,} "
        f"({revenue_outliers.sum()/len(df)*100:.1f}%)"
    )

    # Quantity outliers
    qty_outliers = detect_outliers_iqr(df["quantity"], factor=1.5)
    logger.info(
        f"Quantity outliers (IQR): {qty_outliers.sum():,} of {len(df):,} "
        f"({qty_outliers.sum()/len(df)*100:.1f}%)"
    )

    # ===================================================================
    # PHASE 6: Visualization
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 6: VISUALIZATION")
    logger.info("=" * 70)

    generate_all_plots(
        cohort_sizes=cohort_sizes,
        activity_rates=activity_rates,
        classic_ret=classic_ret,
        rolling_ret=rolling_ret,
        revenue_matrix=revenue_matrix,
        quantity_matrix=quantity_matrix,
        clv_df=clv,
        output_dir=args.output,
    )

    # ===================================================================
    # SUMMARY
    # ===================================================================
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("  ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total rows analyzed:      {len(df):,}")
    logger.info(f"  Unique customers:         {df['customer_id'].nunique():,}")
    logger.info(f"  Cohorts identified:        {len(cohort_sizes)}")
    logger.info(f"  Plots generated:           10")
    logger.info(f"  Output directory:          {args.output}/")
    logger.info(f"  Elapsed time:              {elapsed:.1f}s")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
