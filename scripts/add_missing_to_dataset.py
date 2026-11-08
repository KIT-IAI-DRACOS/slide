"""
Mask exogenous columns jointly and fill them with their full-file mean.

One Bernoulli draw per row is shared by every column in ``--columns``.
A ``missing`` indicator is appended (1 = masked, 0 = kept). Raise
``num_future_features`` by one when pointing a dataset config at the output.

    uv run python scripts/add_missing_to_dataset.py \\
        --input data/epf/DE.csv \\
        --output data/epf/DE_015_missing.csv \\
        --columns "Ampirion Load Forecast" "PV+Wind Forecast" \\
        --prob 0.15 --seed 0
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def add_missing(
    df: pd.DataFrame,
    columns: list[str],
    prob: float,
    missing_col: str = "missing",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Mask all ``columns`` together with probability ``prob`` at each row.

    Masked entries are set to the mean of that column on the full input
    frame. ``missing_col`` is 1 on masked rows and 0 elsewhere.
    """
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    if rng is None:
        rng = np.random.default_rng()

    out = df.copy()
    mask = rng.random(size=len(df)) < prob
    out[missing_col] = mask.astype(int)

    for col in columns:
        fill_value = float(df[col].mean())
        out.loc[mask, col] = fill_value
        logger.info("  %-40s  mean = %.6f", col, fill_value)

    n_missing = int(mask.sum())
    logger.info(
        "Masked %d / %d rows (actual rate = %.4f, target p = %.4f).",
        n_missing,
        len(df),
        n_missing / len(df),
        prob,
    )
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Replace exogenous columns with their full-file mean at random "
            "time steps, and append a binary missing indicator."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV.")
    parser.add_argument("--output", "-o", required=True, help="Output CSV.")
    parser.add_argument(
        "--columns",
        "-c",
        nargs="+",
        required=True,
        help="Exogenous columns to mask together. Do not include the target.",
    )
    parser.add_argument(
        "--prob",
        "-p",
        type=float,
        required=True,
        help="Probability that a time step is masked.",
    )
    parser.add_argument(
        "--missing-col",
        default="missing",
        help="Name of the indicator column (default: missing).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Mask seed.")
    parser.add_argument(
        "--date-col",
        default=None,
        help="Time column. Detected from 'date' or 'Date' when omitted.",
    )
    return parser.parse_args(argv)


def load_csv(path: str, date_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col is None:
        if "date" in df.columns:
            date_col = "date"
        elif "Date" in df.columns:
            date_col = "Date"

    if date_col is not None and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        logger.info("Using '%s' as the time index.", date_col)
    else:
        logger.info("No date or Date column found; keeping the default index.")
    return df


def main(argv=None):
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    df = load_csv(str(input_path), args.date_col)
    logger.info("Loaded %d rows x %d columns.", *df.shape)

    unknown = [col for col in args.columns if col not in df.columns]
    if unknown:
        logger.error(
            "Column(s) not found: %s. Available columns: %s",
            unknown,
            list(df.columns),
        )
        sys.exit(1)
    if args.missing_col in df.columns:
        logger.warning("Column '%s' already exists and will be overwritten.", args.missing_col)

    rng = np.random.default_rng(args.seed)
    logger.info(
        "Joint mean fill: p=%.4f, %d column(s), seed=%s.",
        args.prob,
        len(args.columns),
        args.seed,
    )
    out = add_missing(
        df,
        columns=args.columns,
        prob=args.prob,
        missing_col=args.missing_col,
        rng=rng,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path)
    logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
