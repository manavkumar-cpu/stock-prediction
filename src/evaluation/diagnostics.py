import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import PROJECT_ROOT
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.diagnostics")


def summarize_groups(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []

    for group_name, group in data.groupby(
        group_column, observed=True, sort=True
    ):
        actual = group["Actual_Close"].to_numpy(dtype=float)

        for model in ["Naive", "LSTM"]:
            predicted = group[f"{model}_Prediction"].to_numpy(dtype=float)
            error = predicted - actual

            rows.append({
                "Group": str(group_name),
                "Model": model,
                "Samples": len(group),
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(error ** 2))),
                "MAPE_percent": float(
                    100 * np.mean(np.abs(error / actual))
                ),
                "Mean_Error": float(np.mean(error)),
                "Underprediction_percent": float(
                    100 * np.mean(error < 0)
                ),
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    output_dir = run_dir / "evaluation"

    with (run_dir / "config.json").open(
        "r", encoding="utf-8"
    ) as file:
        config = json.load(file)

    data = pd.read_csv(
        output_dir / "validation_predictions.csv",
        parse_dates=["Forecast_Date", "Target_Date"],
    )

    required = [
        "Forecast_Date",
        "Target_Date",
        "Actual_Close",
        "Naive_Prediction",
        "LSTM_Prediction",
    ]

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    if data.empty or data[required].isna().any().any():
        raise DataValidationError("Prediction table is empty or has missing values.")

    prices = data[
        ["Actual_Close", "Naive_Prediction", "LSTM_Prediction"]
    ].to_numpy(dtype=float)

    if not np.isfinite(prices).all():
        raise DataValidationError("Prices contain infinite values.")

    if (data[["Actual_Close", "Naive_Prediction"]] <= 0).any().any():
        raise DataValidationError("Actual or forecast-date prices are nonpositive.")

    dates = data["Target_Date"]

    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise DataValidationError("Target dates must be unique and chronological.")

    within_validation = (
        (dates > pd.Timestamp(config["split"]["train_end"]))
        & (dates <= pd.Timestamp(config["split"]["validation_end"]))
    )

    if not within_validation.all():
        raise DataValidationError("Found targets outside the validation period.")

    if not (data["Forecast_Date"] < dates).all():
        raise DataValidationError("Forecast dates must precede target dates.")

    data["Year"] = dates.dt.year

    # Naive_Prediction equals the Close available at forecast time.
    # Fixed, descriptive bands—not optimized model thresholds.
    data["Forecast_Price_Band"] = pd.cut(
        data["Naive_Prediction"],
        bins=[0, 75, 100, 125, 150, np.inf],
        labels=[
            "0 < Close <= 75",
            "75 < Close <= 100",
            "100 < Close <= 125",
            "125 < Close <= 150",
            "Close > 150",
        ],
        right=True,
    )

    yearly = summarize_groups(data, "Year")
    by_price = summarize_groups(data, "Forecast_Price_Band")

    yearly.to_csv(output_dir / "errors_by_year.csv", index=False)
    by_price.to_csv(output_dir / "errors_by_price_band.csv", index=False)

    print("\nValidation errors by target year:")
    print(
        yearly.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nValidation errors by forecast-date Close:")
    print(
        by_price.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    logger.info("Saved yearly diagnostics: %s", output_dir / "errors_by_year.csv")
    logger.info(
        "Saved price-band diagnostics: %s",
        output_dir / "errors_by_price_band.csv",
    )


if __name__ == "__main__":
    run_logged(main, logger)