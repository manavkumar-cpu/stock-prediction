import json

import numpy as np
import pandas as pd

from src.data.download import PROJECT_ROOT, load_config
from src.exceptions import ConfigurationError, DataValidationError
from src.features.market import check_daily_index
from src.logger import get_logger, run_logged


logger = get_logger("data.sequences")


def build_sequences(
    inputs: pd.DataFrame,
    targets: pd.DataFrame,
    window_size: int,
):
    check_daily_index(inputs, "Scaled inputs")
    check_daily_index(targets, "Scaled target forecast dates")

    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, int)
        or window_size < 1
    ):
        raise ConfigurationError("window_size must be a positive integer.")

    # Horizon 1: every input row except the final row has a target.
    if not targets.index.equals(inputs.index[:-1]):
        raise DataValidationError(
            "Forecast dates do not match the scaled input dates."
        )

    actual_target_dates = pd.DatetimeIndex(targets["Target_Date"])

    if not actual_target_dates.equals(inputs.index[1:]):
        raise DataValidationError(
            "Targets must be exactly one observed session ahead."
        )

    target_columns = [
        "Target_Scaled", "Target_Close", "Naive_Prediction"
    ]

    if not np.isfinite(
        targets[target_columns].to_numpy(dtype=float)
    ).all():
        raise DataValidationError("Targets contain invalid numeric values.")

    values = inputs.to_numpy(dtype=float)

    if np.isinf(values).any():
        raise DataValidationError("Inputs contain infinite values.")

    windows = []
    labels = []
    records = []
    skipped_history = 0
    skipped_missing = 0

    for position, (forecast_date, row) in enumerate(targets.iterrows()):
        start_position = position - window_size + 1

        if start_position < 0:
            skipped_history += 1
            continue

        # The stop position is exclusive, hence position + 1.
        window = values[start_position:position + 1]

        if not np.isfinite(window).all():
            skipped_missing += 1
            continue

        windows.append(window)
        labels.append(row["Target_Scaled"])

        records.append({
            "Window_Start": inputs.index[start_position],
            "Forecast_Date": forecast_date,
            "Target_Date": row["Target_Date"],
            "Target_Close": row["Target_Close"],
            "Naive_Prediction": row["Naive_Prediction"],
            "Split": row["Split"],
        })

    if not windows:
        raise DataValidationError("No usable sequences were created.")

    X = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    metadata = pd.DataFrame(records)

    if X.shape[1:] != (window_size, inputs.shape[1]):
        raise DataValidationError("Unexpected sequence dimensions.")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise DataValidationError("Invalid values after float32 conversion.")

    if not (
        metadata["Window_Start"] <= metadata["Forecast_Date"]
    ).all():
        raise DataValidationError("Invalid window start dates.")

    if not (
        metadata["Forecast_Date"] < metadata["Target_Date"]
    ).all():
        raise DataValidationError("Target overlaps its input window.")

    logger.info("Skipped for insufficient history: %s", skipped_history)
    logger.info("Skipped for missing input values: %s", skipped_missing)

    return X, y, metadata


def main() -> None:
    config = load_config()
    forecast = config["forecast"]
    experiment = config["experiment"]["name"]

    if forecast["horizon"] != 1:
        raise ConfigurationError("Sequence builder supports horizon: 1.")

    window_size = forecast["window_size"]
    columns = forecast["feature_columns"]

    data_dir = (
        PROJECT_ROOT / config["data"]["processed_dir"] / experiment
    )
    artifact_dir = (
        PROJECT_ROOT / config["paths"]["artifacts"] / experiment
    )

    scaling_metadata = json.loads(
        (artifact_dir / "scaling_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    expected_settings = {
        "experiment": experiment,
        "ticker": config["data"]["ticker"].strip().upper(),
        "feature_columns": columns,
        "target_column": forecast["target_column"],
        "horizon": forecast["horizon"],
        "split": config["split"],
    }

    for key, expected in expected_settings.items():
        if scaling_metadata[key] != expected:
            raise ConfigurationError(
                f"Scaling metadata differs for {key}. "
                "Rebuild the upstream outputs for this configuration."
            )

    inputs = pd.read_csv(
        data_dir / "features_scaled.csv",
        index_col="Date",
        parse_dates=["Date"],
    )
    targets = pd.read_csv(
        data_dir / "targets_scaled.csv",
        index_col="Forecast_Date",
        parse_dates=["Forecast_Date", "Target_Date"],
    )

    if inputs.columns.tolist() != columns:
        raise ConfigurationError(
            "Scaled input columns or their order differ from the config."
        )

    # Verify split labels using target dates.
    dates = targets["Target_Date"]
    train_end = pd.Timestamp(config["split"]["train_end"])
    validation_end = pd.Timestamp(config["split"]["validation_end"])
    test_end = pd.Timestamp(config["split"]["test_end"])

    if dates.isna().any() or (dates > test_end).any():
        raise DataValidationError("Invalid target dates.")

    expected_splits = np.where(
        dates <= train_end,
        "train",
        np.where(dates <= validation_end, "validation", "test"),
    )

    if not np.array_equal(targets["Split"].to_numpy(), expected_splits):
        raise DataValidationError("Incorrect target split assignments.")

    X, y, metadata = build_sequences(inputs, targets, window_size)

    # Check every split before writing outputs.
    partitions = {}

    for split_name in ["train", "validation", "test"]:
        mask = metadata["Split"].eq(split_name).to_numpy()

        if not mask.any():
            raise DataValidationError(
                f"No usable sequences for {split_name}."
            )

        partitions[split_name] = (
            X[mask],
            y[mask],
            metadata.loc[mask].reset_index(drop=True),
        )

    output_dir = data_dir / f"sequences_w{window_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for split_name, (split_X, split_y, split_meta) in partitions.items():
        np.savez_compressed(
            output_dir / f"{split_name}.npz",
            X=split_X,
            y=split_y,
        )

        # Row i corresponds to X[i] and y[i] in this split's NPZ.
        split_meta.index.name = "Sample"
        split_meta.to_csv(output_dir / f"{split_name}_metadata.csv")

        summary_rows.append({
            "Split": split_name,
            "Samples": len(split_meta),
            "First_Target": str(split_meta["Target_Date"].min().date()),
            "Last_Target": str(split_meta["Target_Date"].max().date()),
        })

        logger.info(
            "%s: X shape=%s, y shape=%s",
            split_name,
            split_X.shape,
            split_y.shape,
        )

    sequence_settings = {
        **expected_settings,
        "window_size": window_size,
        "dtype": "float32",
        "input_shape_per_sample": [window_size, len(columns)],
        "summary": summary_rows,
    }

    (output_dir / "sequence_metadata.json").write_text(
        json.dumps(sequence_settings, indent=2),
        encoding="utf-8",
    )

    print("\nSequence summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))

    date_columns = ["Window_Start", "Forecast_Date", "Target_Date"]

    print("\nFirst training sequence dates:")
    print(partitions["train"][2][date_columns].head(1).to_string(index=False))

    print("\nFirst validation sequence dates:")
    print(
        partitions["validation"][2][date_columns]
        .head(1)
        .to_string(index=False)
    )

    logger.info("Sequence date and shape checks: PASS")
    logger.info("Saved sequences and metadata: %s", output_dir)


if __name__ == "__main__":
    run_logged(main, logger)