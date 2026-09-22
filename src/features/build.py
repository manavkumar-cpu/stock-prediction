import numpy as np
import pandas as pd
from src.logger import get_logger, run_logged
from src.data.download import PROJECT_ROOT, load_config
logger = get_logger("features.build")

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder smoothing, initialized with mean gains/losses."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    rsi = pd.Series(np.nan, index=close.index, dtype=float)

    if len(close) <= period:
        return rsi

    # First RSI needs 'period' changes, hence period + 1 prices.
    avg_gain = gains.iloc[1:period + 1].mean()
    avg_loss = losses.iloc[1:period + 1].mean()

    for i in range(period, len(close)):
        if i > period:
            avg_gain = (
                avg_gain * (period - 1) + gains.iloc[i]
            ) / period
            avg_loss = (
                avg_loss * (period - 1) + losses.iloc[i]
            ) / period

        if avg_gain == 0 and avg_loss == 0:
            value = 50.0  # Our convention for completely flat prices.
        elif avg_loss == 0:
            value = 100.0
        else:
            relative_strength = avg_gain / avg_loss
            value = 100 - 100 / (1 + relative_strength)

        rsi.iloc[i] = value

    return rsi


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate features using current and earlier observations only."""
    required = ["Open", "High", "Low", "Close", "Volume"]

    if data.empty:
        raise ValueError("Input data is empty.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex.")

    if (
        data.index.hasnans
        or data.index.has_duplicates
        or not data.index.is_monotonic_increasing
    ):
        raise ValueError("Dates must be valid, unique, and ordered.")

    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    values = data[required].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Required raw values must be finite.")

    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Prices must be positive.")

    if (data["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    features = data.copy()
    close = features["Close"]
    volume = features["Volume"]

    features["Return_1"] = close.pct_change(fill_method=None)
    features["Log_Return_1"] = np.log(close / close.shift(1))

    features["SMA_20"] = close.rolling(
        window=20, min_periods=20
    ).mean()

    features["SMA_50"] = close.rolling(
        window=50, min_periods=50
    ).mean()

    # Recursive EMA initialized from the first Close.
    features["EMA_12"] = close.ewm(
        span=12, adjust=False, min_periods=12
    ).mean()

    features["EMA_26"] = close.ewm(
        span=26, adjust=False, min_periods=26
    ).mean()

    features["MACD"] = features["EMA_12"] - features["EMA_26"]

    features["MACD_Signal"] = features["MACD"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()

    features["MACD_Hist"] = (
        features["MACD"] - features["MACD_Signal"]
    )

    features["RSI_14"] = calculate_rsi(close, period=14)

    # Daily volatility in fractional-return units, not annualized.
    features["Volatility_20"] = features["Return_1"].rolling(
        window=20, min_periods=20
    ).std(ddof=1)

    # Baseline excludes today's volume.
    previous_average_volume = volume.shift(1).rolling(
        window=20, min_periods=20
    ).mean()

    features["Relative_Volume_20"] = (
        volume / previous_average_volume.replace(0, np.nan)
    )

    features["Range_Ratio"] = (
        features["High"] - features["Low"]
    ) / close

    return features


def main() -> None:
    config = load_config()
    settings = config["data"]

    ticker = settings["ticker"].strip().upper()
    start = settings["start_date"]
    end = settings["end_date_exclusive"]

    input_path = (
        PROJECT_ROOT
        / settings["raw_dir"]
        / f"{ticker}_{start}_{end}_raw.csv"
    )

    data = pd.read_csv(
        input_path,
        index_col="Date",
        parse_dates=["Date"],
    )

    features = build_features(data)

    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError("Feature table contains infinite values.")

    # Verify that adding later observations does not change earlier
    # features. Use a cutoff within training for this check.
    prefix = data.loc[:config["split"]["train_end"]].iloc[:200]

    if len(prefix) < 60:
        raise ValueError("Insufficient training history for verification.")

    prefix_features = build_features(prefix)

    pd.testing.assert_frame_equal(
        prefix_features,
        features.loc[prefix.index],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    output_dir = PROJECT_ROOT / settings["processed_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"{ticker}_{start}_{end}_features.csv"
    )

    # Preserve every date, including initial warm-up rows.
    features.to_csv(output_path, index_label="Date")

    # Inspect only training-period feature values.
    train_features = features.loc[:config["split"]["train_end"]]
    new_columns = [
        column for column in features if column not in data.columns
    ]

    missing_counts = train_features[new_columns].isna().sum()

    print("Feature table shape:", features.shape)
    print("Added columns:", new_columns)

    print("\nTraining-period missing counts:")
    print(missing_counts.to_string())

    print("\nFirst complete training feature rows:")
    print(
        train_features.dropna(subset=new_columns)
        .head(3)
        .round(4)
        .to_string()
    )

    logger.info("Prefix consistency check: PASS")
    logger.info(
        "Raw date index preserved: %s",
        features.index.equals(data.index),
    )
    logger.info("Saved feature table: %s", output_path)


if __name__ == "__main__":
    run_logged(main, logger)