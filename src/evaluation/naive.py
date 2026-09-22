import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.download import PROJECT_ROOT, load_config
from src.evaluation.metrics import regression_metrics
from src.exceptions import ConfigurationError, DataValidationError
from src.features.market import check_daily_index
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.naive")


def main() -> None:
    config = load_config()
    settings = config["data"]
    forecast = config["forecast"]

    experiment = config["experiment"]["name"]
    window_size = forecast["window_size"]
    ticker = settings["ticker"].strip().upper()

    if forecast["horizon"] != 1 or forecast["target_column"] != "Close":
        raise ConfigurationError(
            "This baseline implementation expects next-session Close."
        )

    sequence_dir = (
        PROJECT_ROOT
        / settings["processed_dir"]
        / experiment
        / f"sequences_w{window_size}"
    )

    sequence_settings = json.loads(
        (sequence_dir / "sequence_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    expected_settings = {
        "experiment": experiment,
        "ticker": ticker,
        "feature_columns": forecast["feature_columns"],
        "target_column": forecast["target_column"],
        "horizon": forecast["horizon"],
        "window_size": window_size,
        "split": config["split"],
    }

    for key, value in expected_settings.items():
        if sequence_settings[key] != value:
            raise ConfigurationError(
                f"Sequence configuration mismatch: {key}"
            )

    metadata = pd.read_csv(
        sequence_dir / "validation_metadata.csv",
        index_col="Sample",
        parse_dates=["Window_Start", "Forecast_Date", "Target_Date"],
    )

    if metadata.empty or not metadata["Split"].eq("validation").all():
        raise DataValidationError("Expected validation examples only.")

    if not np.array_equal(
        metadata.index.to_numpy(), np.arange(len(metadata))
    ):
        raise DataValidationError("Unexpected sample ordering.")

    dates = pd.DatetimeIndex(metadata["Target_Date"])

    if (
        dates.hasnans
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
    ):
        raise DataValidationError("Invalid validation target dates.")

    train_end = pd.Timestamp(config["split"]["train_end"])
    validation_end = pd.Timestamp(config["split"]["validation_end"])

    if not ((dates > train_end) & (dates <= validation_end)).all():
        raise DataValidationError("Targets fall outside validation.")

    # Cross-check the saved labels and naive predictions.
    start = settings["start_date"]
    end = settings["end_date_exclusive"]

    raw_path = (
        PROJECT_ROOT
        / settings["raw_dir"]
        / f"{ticker}_{start}_{end}_raw.csv"
    )

    raw = pd.read_csv(
        raw_path,
        index_col="Date",
        parse_dates=["Date"],
    ).loc[:validation_end]

    check_daily_index(raw, "Development raw data")

    forecast_dates = pd.DatetimeIndex(metadata["Forecast_Date"])
    forecast_positions = raw.index.get_indexer(forecast_dates)
    target_positions = raw.index.get_indexer(dates)

    if (
        (forecast_positions < 0).any()
        or (target_positions < 0).any()
        or not np.all(target_positions == forecast_positions + 1)
    ):
        raise DataValidationError("Invalid next-session alignment.")

    actual = metadata["Target_Close"].to_numpy(dtype=float)
    predicted = metadata["Naive_Prediction"].to_numpy(dtype=float)

    np.testing.assert_allclose(
        actual,
        raw["Close"].reindex(dates).to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        predicted,
        raw["Close"].reindex(forecast_dates).to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )

    scores = regression_metrics(actual, predicted)

    report = {
        "model": "naive_persistence",
        "experiment": experiment,
        "ticker": ticker,
        "split": "validation",
        "horizon": 1,
        "samples": len(metadata),
        "first_target": str(dates.min().date()),
        "last_target": str(dates.max().date()),
        "price_basis": "Yahoo Close, USD, split-adjusted",
        "metrics": scores,
    }

    metrics_dir = (
        PROJECT_ROOT
        / config["paths"]["metrics"]
        / experiment
        / f"w{window_size}"
    )
    figures_dir = (
        PROJECT_ROOT
        / config["paths"]["figures"]
        / experiment
        / f"w{window_size}"
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    report_path = metrics_dir / "naive_validation_metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    predictions = metadata[
        ["Forecast_Date", "Target_Date"]
    ].copy()
    predictions["Actual_Close"] = actual
    predictions["Predicted_Close"] = predicted

    # Positive error means the prediction was above the actual price.
    predictions["Error"] = predicted - actual
    predictions["Absolute_Error"] = np.abs(predicted - actual)

    predictions.to_csv(
        metrics_dir / "naive_validation_predictions.csv",
        index_label="Sample",
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Full validation period.
    axes[0].plot(dates, actual, label="Actual", linewidth=1)
    axes[0].plot(
        dates, predicted,
        label="Naive prediction",
        linewidth=1,
        alpha=0.75,
    )
    axes[0].set_title(f"{ticker} — Naive validation forecast")

    # Fixed first-60-session view, not selected based on performance.
    axes[1].plot(dates[:60], actual[:60], label="Actual", linewidth=1)
    axes[1].plot(
        dates[:60], predicted[:60],
        label="Naive prediction",
        linewidth=1,
        linestyle="--",
    )
    axes[1].set_title("First 60 validation targets — closer view")

    for ax in axes:
        ax.set_xlabel("Target date")
        ax.set_ylabel("Close (USD, split-adjusted)")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    figure_path = figures_dir / "naive_validation.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    print("\nNaive baseline — validation")
    print("Samples:", len(metadata))
    print("Target period:", dates.min().date(), "to", dates.max().date())

    for name, value in scores.items():
        print(f"{name}: {value:.6f}")

    logger.info("Naive prediction and target alignment checks: PASS")
    logger.info("Saved metrics: %s", report_path)
    logger.info("Saved prediction table in: %s", metrics_dir)
    logger.info("Saved plot: %s", figure_path)


if __name__ == "__main__":
    run_logged(main, logger)