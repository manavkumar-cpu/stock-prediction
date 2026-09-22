import numpy as np
import pandas as pd
from src.logger import get_logger, run_logged
from src.data.download import PROJECT_ROOT, load_config
from src.features.market import check_daily_index
logger=get_logger("data.prepare")
def create_target_table(
    data: pd.DataFrame,
    target_column: str,
    split_settings: dict,
) -> pd.DataFrame:
    check_daily_index(data, "Feature table")

    if target_column not in data.columns:
        raise ValueError(f"Missing target source: {target_column}")

    prices = pd.to_numeric(data[target_column], errors="coerce")

    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("Target source contains missing or infinite values.")

    if (prices <= 0).any():
        raise ValueError("Target prices must be positive.")

    train_end = pd.Timestamp(split_settings["train_end"])
    validation_end = pd.Timestamp(split_settings["validation_end"])
    test_end = pd.Timestamp(split_settings["test_end"])

    if not train_end < validation_end < test_end:
        raise ValueError("Split boundaries must be strictly increasing.")

    targets = pd.DataFrame(index=data.index)
    targets.index.name = "Forecast_Date"

    session_dates = pd.Series(data.index, index=data.index)

    targets["Target_Date"] = session_dates.shift(-1)
    targets["Target_Close"] = prices.shift(-1)

    # For the naive baseline: predict the most recent observed Close.
    targets["Naive_Prediction"] = prices

    # The final observation has no next-session target in this dataset.
    targets = targets.loc[targets["Target_Date"].notna()].copy()

    if targets.empty:
        raise ValueError("No labeled examples available.")

    if not (targets["Target_Date"] > targets.index).all():
        raise ValueError("Every target date must follow its forecast date.")

    targets["Split"] = pd.Series(
        index=targets.index, dtype="string"
    )

    target_dates = targets["Target_Date"]

    targets.loc[
        target_dates <= train_end,
        "Split",
    ] = "train"

    targets.loc[
        (target_dates > train_end)
        & (target_dates <= validation_end),
        "Split",
    ] = "validation"

    targets.loc[
        (target_dates > validation_end)
        & (target_dates <= test_end),
        "Split",
    ] = "test"

    if targets["Split"].isna().any():
        raise ValueError("Some target dates fall outside the split limits.")

    # Verify labels against the actual source prices on target dates.
    expected_prices = prices.reindex(
        pd.DatetimeIndex(targets["Target_Date"])
    ).to_numpy()

    np.testing.assert_array_equal(
        targets["Target_Close"].to_numpy(),
        expected_prices,
    )

    return targets


def main() -> None:
    config = load_config()
    settings = config["data"]
    forecast = config["forecast"]

    # Multi-horizon construction will be implemented separately.
    if forecast["horizon"] != 1:
        raise ValueError("This phase supports horizon: 1 only.")

    if forecast["target_column"] != "Close":
        raise ValueError("This experiment expects target_column: Close.")

    ticker = settings["ticker"].strip().upper()
    start = settings["start_date"]
    end = settings["end_date_exclusive"]

    processed_dir = PROJECT_ROOT / settings["processed_dir"]

    input_path = processed_dir / (
        f"{ticker}_{start}_{end}_features_market.csv"
    )

    features = pd.read_csv(
        input_path,
        index_col="Date",
        parse_dates=["Date"],
    )

    selected_features = forecast["feature_columns"]

    if not selected_features:
        raise ValueError("Select at least one input feature.")

    missing = [
        column for column in selected_features
        if column not in features.columns
    ]
    if missing:
        raise ValueError(f"Selected features are missing: {missing}")

    targets = create_target_table(
        data=features,
        target_column=forecast["target_column"],
        split_settings=config["split"],
    )

    summary_rows = []

    for split_name in ["train", "validation", "test"]:
        subset = targets.loc[targets["Split"] == split_name]

        if subset.empty:
            raise ValueError(f"{split_name} has no labeled examples.")

        summary_rows.append({
            "Split": split_name,
            "Labeled_Rows": len(subset),
            "First_Forecast_Date": subset.index.min().date(),
            "Last_Forecast_Date": subset.index.max().date(),
            "First_Target_Date": subset["Target_Date"].min().date(),
            "Last_Target_Date": subset["Target_Date"].max().date(),
        })

    summary = pd.DataFrame(summary_rows)

    output_path = processed_dir / (
        f"{ticker}_{start}_{end}_targets_h1.csv"
    )
    targets.to_csv(output_path, index_label="Forecast_Date")

    metrics_dir = PROJECT_ROOT / config["paths"]["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_path = metrics_dir / f"{ticker}_target_split_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Selected input features:", selected_features)
    print("Source observations:", len(features))
    print("Labeled rows before windowing:", len(targets))
    print("Rows without a future target:", len(features) - len(targets))

    print("\nSplit summary:")
    print(summary.to_string(index=False))

    print("\nFirst three training examples:")
    print(
        targets.loc[targets["Split"] == "train"]
        .head(3)
        .to_string()
    )

    logger.info("Target alignment check: PASS")
    logger.info("Saved targets: %s", output_path)
    logger.info("Saved split summary: %s", summary_path)

if __name__ == "__main__":
    run_logged(main, logger)