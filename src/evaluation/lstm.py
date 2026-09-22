import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.download import PROJECT_ROOT
from src.evaluation.metrics import regression_metrics
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.lstm")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataValidationError(message)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training run folder containing best.keras and saved scalers.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir = run_dir.resolve()

    # Use the configuration recorded when this model was trained.
    config = read_json(run_dir / "config.json")
    forecast = config["forecast"]

    require(
        forecast["feature_columns"] == ["Close"]
        and forecast["target_column"] == "Close"
        and forecast["horizon"] == 1,
        "This evaluator currently supports Close-only, one-session forecasts.",
    )

    experiment = config["experiment"]["name"]
    window_size = forecast["window_size"]

    sequence_dir = (
        PROJECT_ROOT
        / config["data"]["processed_dir"]
        / experiment
        / f"sequences_w{window_size}"
    )

    # Detect changes to sequence settings since training.
    require(
        read_json(sequence_dir / "sequence_metadata.json")
        == read_json(run_dir / "sequence_metadata.json"),
        "Sequence metadata differs from the training run.",
    )

    with np.load(
        sequence_dir / "validation.npz", allow_pickle=False
    ) as arrays:
        X = arrays["X"]
        y_scaled = arrays["y"]

    metadata = pd.read_csv(
        sequence_dir / "validation_metadata.csv",
        index_col=0,
        parse_dates=["Window_Start", "Forecast_Date", "Target_Date"],
    )

    n = len(metadata)

    require(n > 1, "Not enough validation samples.")
    require(
        X.shape == (n, window_size, 1),
        f"Unexpected validation input shape: {X.shape}",
    )
    require(
        y_scaled.shape == (n, 1),
        f"Unexpected validation target shape: {y_scaled.shape}",
    )
    require(
        np.isfinite(X).all() and np.isfinite(y_scaled).all(),
        "Validation arrays contain missing or infinite values.",
    )
    require(
        np.array_equal(metadata.index.to_numpy(), np.arange(n)),
        "Validation sample order is inconsistent.",
    )
    require(
        metadata["Split"].eq("validation").all(),
        "Metadata contains observations outside the validation split.",
    )

    dates = metadata["Target_Date"]
    require(
        dates.notna().all()
        and dates.is_unique
        and dates.is_monotonic_increasing,
        "Target dates must be valid, unique, and chronological.",
    )
    require(
        (
            (dates > pd.Timestamp(config["split"]["train_end"]))
            & (dates <= pd.Timestamp(config["split"]["validation_end"]))
        ).all(),
        "Target dates fall outside the validation period.",
    )
    require(
        (metadata["Window_Start"] <= metadata["Forecast_Date"]).all()
        and (metadata["Forecast_Date"] < dates).all(),
        "Window, forecast, and target dates are inconsistent.",
    )

    # Load only the trusted scaler files created by this project.
    with (run_dir / "y_scaler.pkl").open("rb") as file:
        y_scaler = pickle.load(file)

    with (run_dir / "x_scaler.pkl").open("rb") as file:
        x_scaler = pickle.load(file)

    actual = metadata["Target_Close"].to_numpy(dtype=float)
    naive = metadata["Naive_Prediction"].to_numpy(dtype=float)

    restored_targets = y_scaler.inverse_transform(
        y_scaled.astype(np.float64)
    ).ravel()

    require(
        np.allclose(
            restored_targets, actual, rtol=1e-6, atol=1e-5
        ),
        "Inverse-scaled targets do not match metadata prices.",
    )

    # For Close-only inputs, the final input is today's Close.
    restored_last_inputs = x_scaler.inverse_transform(
        X[:, -1, :].astype(np.float64)
    ).ravel()

    require(
        np.allclose(
            restored_last_inputs, naive, rtol=1e-6, atol=1e-5
        ),
        "Last input prices do not match naive predictions.",
    )

    model = tf.keras.models.load_model(
        run_dir / "best.keras",
        compile=False,
    )

    require(
        tuple(model.input_shape[1:]) == tuple(X.shape[1:]),
        "Model input shape does not match validation sequences.",
    )

    predictions_scaled = model.predict(
        X,
        batch_size=config["training"]["batch_size"],
        verbose=0,
    )

    require(
        predictions_scaled.shape == (n, 1)
        and np.isfinite(predictions_scaled).all(),
        "Model predictions have an invalid shape or nonfinite values.",
    )

    predicted = y_scaler.inverse_transform(
        predictions_scaled.astype(np.float64)
    ).ravel()

    # Both models use exactly the same actual prices and dates.
    lstm_metrics = regression_metrics(actual, predicted)
    naive_metrics = regression_metrics(actual, naive)

    comparison = pd.DataFrame(
        {"Naive": naive_metrics, "LSTM": lstm_metrics}
    ).T
    comparison.index.name = "Model"

    predictions = metadata[
        ["Window_Start", "Forecast_Date", "Target_Date"]
    ].copy()
    predictions["Actual_Close"] = actual
    predictions["Naive_Prediction"] = naive
    predictions["LSTM_Prediction"] = predicted
    predictions["Naive_Error"] = naive - actual
    predictions["LSTM_Error"] = predicted - actual
    predictions["LSTM_Absolute_Error"] = np.abs(predicted - actual)

    improvements = {
        metric: float(
            100 * (1 - lstm_metrics[metric] / naive_metrics[metric])
        )
        for metric in ["MAE", "RMSE"]
        if naive_metrics[metric] > 0
    }

    report = {
        "run_id": run_dir.name,
        "split": "validation",
        "samples": n,
        "first_target": dates.min().strftime("%Y-%m-%d"),
        "last_target": dates.max().strftime("%Y-%m-%d"),
        "metrics": {
            "Naive": naive_metrics,
            "LSTM": lstm_metrics,
        },
        "lstm_mean_error_usd": float(np.mean(predicted - actual)),
        "improvement_over_naive_percent": improvements,
    }

    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_dir / "validation_comparison.csv")
    predictions.to_csv(
        output_dir / "validation_predictions.csv", index=False
    )
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    axes[0].plot(dates, actual, label="Actual", color="black", linewidth=1)
    axes[0].plot(dates, naive, label="Naive", alpha=0.6, linewidth=1)
    axes[0].plot(dates, predicted, label="LSTM", linewidth=1)
    axes[0].set_title("AAPL — Validation forecasts")
    axes[0].set_ylabel("Close (USD, split-adjusted)")
    axes[0].legend()

    axes[1].plot(
        dates, naive - actual, label="Naive error", alpha=0.6
    )
    axes[1].plot(dates, predicted - actual, label="LSTM error")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Prediction − actual (USD)")
    axes[1].set_xlabel("Target date")
    axes[1].legend()

    for axis in axes:
        axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "validation_predictions.png", dpi=160)
    plt.close(fig)

    print("\nValidation comparison:")
    print(comparison.to_string(float_format=lambda value: f"{value:.6f}"))

    print(f"\nLSTM mean error: ${report['lstm_mean_error_usd']:.4f}")

    for metric, improvement in improvements.items():
        print(f"{metric} improvement over naive: {improvement:.2f}%")

    print("\nTen largest LSTM absolute errors:")
    columns = [
        "Target_Date", "Actual_Close", "LSTM_Prediction",
        "Naive_Prediction", "LSTM_Error",
    ]
    print(
        predictions.nlargest(10, "LSTM_Absolute_Error")[columns]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    logger.info("Validation alignment and inverse-scaling checks: PASS")
    logger.info("Saved evaluation outputs: %s", output_dir)


if __name__ == "__main__":
    run_logged(main, logger)