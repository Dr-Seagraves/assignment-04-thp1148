"""
Assignment 04: Simple Linear Regression (REIT Annual Returns and Predictors)

This script estimates three simple OLS regressions of REIT annual returns on different
X variables:
- ret (annual) ~ div12m_me: dividend yield (12-month)
- ret (annual) ~ prime_rate: prime loan rate (interest rate sensitivity)
- ret (annual) ~ ffo_at_reit: FFO to assets (fundamental performance)

Data: Pre-built annual REIT dataset (REIT_sample_annual_*.csv), one observation
per REIT per year. Interest rates merged from interest_rates_monthly.csv (December values).

Learning objectives:
- Fit simple linear regressions using statsmodels
- Extract and interpret regression coefficients
- Visualize relationships with scatter plots
- Save results for the interpretation memo
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

try:
    from config_paths import DATA_DIR, RESULTS_DIR
except ImportError:
    _ROOT = Path(__file__).resolve().parent
    DATA_DIR = _ROOT / "data"
    RESULTS_DIR = _ROOT / "Results"
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

# Three regressions: (x_var, title, xlabel)
REGRESSION_SPECS = [
    ("div12m_me", "REIT Annual Return vs. Dividend Yield", "Dividend Yield (12m)"),
    ("prime_rate", "REIT Annual Return vs. Prime Loan Rate", "Prime Loan Rate (%)"),
    ("ffo_at_reit", "REIT Annual Return vs. FFO to Assets", "FFO / Assets"),
]


def _merge_annual_interest_rates(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """
    Merge year-end (December) interest rates by year.
    Run fetch_interest_rates.py first to create interest_rates_monthly.csv.
    """
    rates_path = data_dir / "interest_rates_monthly.csv"
    if not rates_path.exists():
        raise FileNotFoundError(
            f"Missing {rates_path}\n"
            "Run fetch_interest_rates.py to download interest rate data."
        )
    rates = pd.read_csv(rates_path)
    rates["date"] = pd.to_datetime(rates["date"])
    rates["year"] = rates["date"].dt.year
    rates["month"] = rates["date"].dt.month
    rates_dec = rates[rates["month"] == 12][["year", "mortgage_30y", "prime_rate"]].copy()

    df = df.copy()
    if "year" not in df.columns and "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df = df.merge(rates_dec, on="year", how="left")
    return df


def load_reit_annual_data(data_path: Path) -> pd.DataFrame:
    """
    Load pre-built annual REIT data (REIT_sample_annual_*.csv).

    Steps:
    1. Read the CSV file.
    2. If 'year' is missing but 'date' exists, extract year from date.
    3. Rename ret12 to ret (annual return = 12-month cumulative return).
    4. Merge year-end interest rates using _merge_annual_interest_rates.
    5. Drop rows with missing ret, div12m_me, or prime_rate.

    Note: ffo_at_reit may have missing values; the third regression (ret ~ ffo_at_reit)
    will automatically drop those rows and may report a smaller N than the first two.

    Parameters
    ----------
    data_path : Path
        Path to the REIT annual CSV file.

    Returns
    -------
    pd.DataFrame
        REIT data with columns: ret, div12m_me, prime_rate, ffo_at_reit (and others)
    """
    df = pd.read_csv(data_path)

    if "year" not in df.columns and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

    df = df.rename(columns={"ret12": "ret"})

    df = _merge_annual_interest_rates(df, data_path.parent)

    # TODO: Drop rows with missing values in ret, div12m_me, prime_rate
    # Hint: df = df.dropna(subset=["ret", "div12m_me", "prime_rate"])
    df = df.dropna(subset=["ret", "div12m_me", "prime_rate"])

    return df


def estimate_regression(df: pd.DataFrame, x_var: str):
    """
    Estimate OLS regression: ret ~ x_var

    Parameters
    ----------
    df : pd.DataFrame
        REIT data with 'ret' and x_var columns.
    x_var : str
        Name of the X variable (e.g., 'div12m_me', 'prime_rate', 'ffo_at_reit').

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted regression model.
    """
    # Estimate ret ~ x_var using the formula interface.
    model = ols(f"ret ~ {x_var}", data=df).fit()
    return model


def save_regression_summary(model, output_path: Path) -> None:
    """
    Save the regression summary to a text file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(str(model.summary()))


def plot_scatter_with_regression(
    df: pd.DataFrame, model, x_var: str, title: str, xlabel: str, output_path: Path
) -> None:
    """
    Create a scatter plot with the fitted regression line.

    Tips:
    - Use only rows with valid x_var and ret (dropna)
    - Scatter: x=x_var, y='ret' (use alpha for transparency)
    - Regression line: compute y = intercept + slope * x over a range of x values
    - Zoom axis limits to central data (e.g., 2nd–98th percentiles) so the slope is easier to see
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    df_valid = df[["ret", x_var]].dropna()
    x = df_valid[x_var]
    y = df_valid["ret"]

    ax.scatter(x, y, alpha=0.6, label="Observations")

    intercept = model.params["Intercept"]
    slope = model.params[x_var]
    x_min, x_max = np.percentile(x, [2, 98])
    x_line = np.linspace(x_min, x_max, 200)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color="black", linewidth=2, label="Fitted line")

    y_min, y_max = np.percentile(y, [2, 98])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    r2 = model.rsquared
    ax.set_title(f"{title} (R2 = {r2:.3f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Annual Return")
    ax.legend()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_key_results(model, x_var: str) -> None:
    """
    Print key regression results to the console.
    """
    print("\n" + "=" * 60)
    print(f"ret (annual) ~ {x_var.upper()}")
    print("=" * 60)
    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues

    print("Coefficients:")
    print(
        f"  Intercept: {params['Intercept']:.6f} | SE {bse['Intercept']:.6f} | "
        f"t {tvals['Intercept']:.3f} | p {pvals['Intercept']:.4f}"
    )
    print(
        f"  {x_var}: {params[x_var]:.6f} | SE {bse[x_var]:.6f} | "
        f"t {tvals[x_var]:.3f} | p {pvals[x_var]:.4f}"
    )

    print(f"R2: {model.rsquared:.4f} | Adj R2: {model.rsquared_adj:.4f} | N: {int(model.nobs)}")

    slope_sign = "positive" if params[x_var] > 0 else "negative"
    sig_text = "significant" if pvals[x_var] < 0.05 else "not significant"
    print(f"Slope is {slope_sign} and {sig_text} at 5%.")
    print("=" * 60 + "\n")


def _find_reit_data(data_dir: Path) -> Path:
    """Find REIT_sample_annual_*.csv in data folder (uses most recently modified)."""
    candidates = list(data_dir.glob("REIT_sample_annual_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No REIT_sample_annual_*.csv found in {data_dir}\n"
            "Use the pre-built annual dataset: REIT_sample_annual_2004_2024.csv"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    """Main execution function."""
    data_dir = DATA_DIR
    data_path = _find_reit_data(data_dir)
    results_dir = RESULTS_DIR

    print("Loading REIT annual data (REIT_sample_annual_*.csv)...")
    df = load_reit_annual_data(data_path)
    print(f"Loaded {len(df)} firm-year observations.")
    print(f"Years: {df['year'].min():.0f}–{df['year'].max():.0f}")

    for x_var, title, xlabel in REGRESSION_SPECS:
        print(f"\nEstimating regression: ret (annual) ~ {x_var}")
        model = estimate_regression(df, x_var)
        print_key_results(model, x_var)

        summary_path = results_dir / f"regression_{x_var}.txt"
        plot_path = results_dir / f"scatter_{x_var}.png"
        save_regression_summary(model, summary_path)
        print(f"Saved: {summary_path}")
        plot_scatter_with_regression(df, model, x_var, title, xlabel, plot_path)
        print(f"Saved: {plot_path}")

    print("\n✓ Assignment 04 complete!")
    print("Next steps:")
    print("  1. Review regression outputs in Results/")
    print("  2. Compare coefficients across dividend yield, prime rate, and FFO/Assets")
    print("  3. Complete assignment04_report.md with your interpretation")
    print("  4. Complete AI_AUDIT_APPENDIX.md")


if __name__ == "__main__":
    main()
