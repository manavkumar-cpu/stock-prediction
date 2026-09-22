import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged
from src.data.download import (
    PROJECT_ROOT,
    download_stock_data,
    load_config,
)
logger = get_logger("features.market")

def check_daily_index(data: pd.DataFrame, name: str) -> None:
    if data.empty:
        raise ValueError(f"{name}: dataset is empty.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: expected a DatetimeIndex.")

    if data.index.hasnans or data.index.has_duplicates:
        raise ValueError(f"{name}: invalid or duplicate dates.")

    if not data.index.is_monotonic_increasing:
        raise ValueError(f"{name}: dates are not chronological.")

    if data.index.tz is not None:
        raise ValueError(f"{name}: expected timezone-free daily dates.")

    if not data.index.equals(data.index.normalize()):
        raise ValueError(f"{name}: unexpected time-of-day values.")


def align_market_close(
    data: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    name: str,
) -> pd.Series:
    check_daily_index(data, name)

    if "Close" not in data.columns:
        raise ValueError(f"{name}: Close column is missing.")

    # Align to AAPL sessions before calculating changes or lagging.
    close = pd.to_numeric(
        data["Close"], errors="coerce"
    ).reindex(sessions)

    invalid = close.isna() | ~np.isfinite(close) | (close <= 0)

    if invalid.any():
        dates = close.index[invalid].strftime("%Y-%m-%d").tolist()

        raise DataValidationError(
            f"{name}: missing or invalid Close on "
            f"{len(dates)} AAPL sessions. First dates: {dates[:10]}"
        )

    return close


def build_market_features(
    closes: pd.DataFrame,
    lag_sessions: int,
) -> pd.DataFrame:
    if lag_sessions < 1:
        raise ValueError("This design requires at least one session of lag.")

    context = pd.DataFrame(index=closes.index)

    context["SP500_Return_1"] = closes["SP500"].pct_change(
        fill_method=None
    )
    context["NASDAQ_Return_1"] = closes["NASDAQ"].pct_change(
        fill_method=None
    )
    context["VIX_Level"] = closes["VIX"]
    context["VIX_Change_1"] = closes["VIX"].pct_change(
        fill_method=None
    )

    # Shift values along AAPL's session index.
    context = context.shift(lag_sessions)

    return context.add_suffix(f"_Lag{lag_sessions}")


def main() -> None:
    config = load_config()
    settings = config["data"]
    market_settings = config["market"]

    ticker = settings["ticker"].strip().upper()
    start = settings["start_date"]
    end = settings["end_date_exclusive"]
    lag = market_settings["lag_sessions"]

    processed_dir = PROJECT_ROOT / settings["processed_dir"]
    feature_path = processed_dir / (
        f"{ticker}_{start}_{end}_features.csv"
    )

    features = pd.read_csv(
        feature_path,
        index_col="Date",
        parse_dates=["Date"],
    )
    check_daily_index(features, ticker)

    market_dir = PROJECT_ROOT / settings["raw_dir"] / "market"
    market_dir.mkdir(parents=True, exist_ok=True)

    closes = pd.DataFrame(index=features.index)

    for name in ["SP500", "NASDAQ", "VIX"]:
        symbol = market_settings["tickers"][name]

        cache_path = market_dir / (
            f"{name}_{symbol}_{start}_{end}_raw.csv"
        )
        downloaded = not cache_path.exists()

        if downloaded:
            logger.info("Downloading %s: %s", name, symbol)

            market_data = download_stock_data(
                ticker=symbol,
                start_date=start,
                end_date=end,
                interval="1d",
                auto_adjust=False,
            )
        else:
            logger.info("Using saved data: %s", name)

            market_data = pd.read_csv(
                cache_path,
                index_col="Date",
                parse_dates=["Date"],
            )

        closes[name] = align_market_close(
            market_data,
            features.index,
            name,
        )

        # Cache only after required Close values pass validation.
        if downloaded:
            market_data.to_csv(cache_path, index_label="Date")

        logger.info("%s: coverage check PASS", name)

    context = build_market_features(closes, lag)

    # Early NaNs are expected; later NaNs are not.
    if context.iloc[lag + 1:].isna().any().any():
        raise ValueError("Unexpected missing market features after warm-up.")

    if np.isinf(context.to_numpy(dtype=float)).any():
        raise ValueError("Infinite market feature values found.")

    # Adding later rows must not change earlier market features.
    prefix = closes.loc[:config["split"]["train_end"]].iloc[:200]

    pd.testing.assert_frame_equal(
        build_market_features(prefix, lag),
        context.loc[prefix.index],
    )

    combined = features.join(
        context,
        how="left",
        validate="one_to_one",
    )

    if not combined.index.equals(features.index):
        raise ValueError("AAPL session index changed during the join.")

    output_path = processed_dir / (
        f"{ticker}_{start}_{end}_features_market.csv"
    )
    combined.to_csv(output_path, index_label="Date")

    # Display development observations only.
    train_context = context.loc[:config["split"]["train_end"]]

    print("\nCombined shape:", combined.shape)
    print("Added market columns:", context.columns.tolist())

    print("\nTraining-period missing counts:")
    print(train_context.isna().sum().to_string())

    print("\nFirst five complete training market rows:")
    print(train_context.dropna().head().round(6).to_string())

    logger.info("Market prefix consistency check: PASS")
    logger.info("AAPL dates preserved")
    logger.info("Saved combined features: %s", output_path)


if __name__ == "__main__":
    run_logged(main, logger)