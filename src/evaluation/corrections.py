import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import PROJECT_ROOT
from src.evaluation.metrics import regression_metrics
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.corrections")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    with (run_dir / "config.json").open(
        "r", encoding="utf-8"
    ) as file:
        config = json.load(file)

    sequence_dir = (
        PROJECT_ROOT
        / config["data"]["processed_dir"]
        / config["experiment"]["name"]
        / f"sequences_w{config['forecast']['window_size']}"
    )

    train = pd.read_csv(
        sequence_dir / "train_metadata.csv",
        parse_dates=["Target_Date"],
    )

    output_dir = run_dir / "evaluation"
    validation = pd.read_csv(
        output_dir / "validation_predictions.csv",
        parse_dates=["Target_Date"],
    )

    if train.empty or validation.empty:
        raise DataValidationError("Training or validation table is empty.")

    train_dates = train["Target_Date"]
    validation_dates = validation["Target_Date"]

    if (
        train_dates.isna().any()
        or not train_dates.is_unique
        or not train["Split"].eq("train").all()
        or not (
            train_dates <= pd.Timestamp(config["split"]["train_end"])
        ).all()
    ):
        raise DataValidationError("Invalid training split or target dates.")

    if (
        validation_dates.isna().any()
        or not validation_dates.is_unique
        or not (
            (validation_dates > pd.Timestamp(config["split"]["train_end"]))
            & (
                validation_dates
                <= pd.Timestamp(config["split"]["validation_end"])
            )
        ).all()
    ):
        raise DataValidationError("Invalid validation target dates.")

    train_values = train[
        ["Target_Close", "Naive_Prediction"]
    ].to_numpy(dtype=float)

    validation_values = validation[
        ["Actual_Close", "Naive_Prediction", "LSTM_Prediction"]
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(train_values).all()
        or not np.isfinite(validation_values).all()
    ):
        raise DataValidationError("Found missing or infinite prices.")

    # Estimate the baseline correction using training labels only.
    training_changes = (
        train["Target_Close"] - train["Naive_Prediction"]
    )
    drift = float(training_changes.mean())

    actual = validation["Actual_Close"].to_numpy(dtype=float)
    today = validation["Naive_Prediction"].to_numpy(dtype=float)
    lstm = validation["LSTM_Prediction"].to_numpy(dtype=float)

    fixed_prediction = today + drift
    corrections = lstm - today

    comparison = pd.DataFrame({
        "Naive": regression_metrics(actual, today),
        "Training_mean_change": regression_metrics(
            actual, fixed_prediction
        ),
        "Residual_LSTM": regression_metrics(actual, lstm),
    }).T
    comparison.index.name = "Model"

    correction_summary = {
        "training_samples": len(train),
        "validation_samples": len(validation),
        "training_mean_change_usd": drift,
        "lstm_correction_mean_usd": float(corrections.mean()),
        "lstm_correction_std_usd": float(corrections.std(ddof=0)),
        "lstm_correction_min_usd": float(corrections.min()),
        "lstm_correction_max_usd": float(corrections.max()),
        "lstm_positive_corrections_percent": float(
            100 * np.mean(corrections > 0)
        ),
    }

    comparison.to_csv(output_dir / "constant_change_comparison.csv")

    (output_dir / "correction_summary.json").write_text(
        json.dumps(correction_summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nCorrection summary:")
    print(json.dumps(correction_summary, indent=2))

    print("\nValidation comparison:")
    print(
        comparison.to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )

    logger.info("Constant correction estimated from training data only")
    logger.info("Saved correction analysis: %s", output_dir)


if __name__ == "__main__":
    run_logged(main, logger)