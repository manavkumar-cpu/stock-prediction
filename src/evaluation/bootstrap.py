import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import PROJECT_ROOT
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.bootstrap")

BLOCK_LENGTHS = (5, 10, 20)
REPETITIONS = 5000
SEED = 42


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def circular_indices(groups, block_length, rng):
    """Resample within each year, preserving its original sample count."""
    sampled = []

    for positions in groups:
        n = len(positions)
        block_count = (n + block_length - 1) // block_length
        starts = rng.integers(0, n, size=block_count)

        local_indices = (
            starts[:, None] + np.arange(block_length)[None, :]
        ) % n

        sampled.append(positions[local_indices.ravel()[:n]])

    return np.concatenate(sampled)


def metric_differences(errors):
    """Column zero is naive; subsequent columns are learned models."""
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))

    return np.stack([
        mae[1:] - mae[0],
        rmse[1:] - rmse[0],
    ], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    run_dir = run_dir.resolve()

    experiment = json.loads(
        (run_dir / "experiment.json").read_text(encoding="utf-8")
    )
    horizon = int(experiment["config"]["forecast"]["horizon"])

    require(horizon in (1, 5), "Expected horizon 1 or 5.")

    predictions = pd.read_csv(run_dir / "all_predictions.csv")

    required = [
        "Forecast_Date",
        "Target_Date",
        "Actual_Close",
        "Naive_Prediction",
    ]
    require(
        all(column in predictions.columns for column in required),
        "Prediction table is missing required columns.",
    )

    for column in ("Forecast_Date", "Target_Date"):
        predictions[column] = pd.to_datetime(
            predictions[column], errors="raise"
        )

    dates = predictions["Target_Date"]
    require(
        predictions[["Forecast_Date", "Target_Date"]].notna().all().all(),
        "Missing forecast or target dates.",
    )
    require(
        dates.is_unique and dates.is_monotonic_increasing,
        "Target dates must be unique and chronological.",
    )
    require(
        (predictions["Forecast_Date"] < dates).all(),
        "Forecast dates must precede target dates.",
    )

    years = dates.dt.year.to_numpy()
    expected_counts = (
        {2020: 253, 2021: 252, 2022: 251}
        if horizon == 1
        else {2020: 249, 2021: 248, 2022: 247}
    )

    require(
        set(years) == set(expected_counts),
        "Only development target years 2020–2022 are supported.",
    )

    groups = []
    for year, expected in expected_counts.items():
        positions = np.flatnonzero(years == year)
        require(
            len(positions) == expected,
            f"Unexpected sample count for {year}.",
        )
        groups.append(positions)

    models = [
        name for name in ("Six_feature_LSTM", "Attention_LSTM")
        if name in predictions.columns
    ]
    require(bool(models), "No supported LSTM prediction columns found.")

    actual = predictions["Actual_Close"].to_numpy(dtype=float)
    forecasts = predictions[
        ["Naive_Prediction"] + models
    ].to_numpy(dtype=float)

    require(
        np.isfinite(actual).all() and np.isfinite(forecasts).all(),
        "Actual values or predictions contain nonfinite values.",
    )
    require((actual > 0).all(), "Actual prices must be positive.")

    # The same row index always selects actual, naive, and all models.
    errors = forecasts - actual[:, None]
    observed = metric_differences(errors)

    # An exact copy of naive must have zero metric differences.
    identity_errors = np.column_stack([errors[:, 0], errors[:, 0]])
    require(
        np.allclose(metric_differences(identity_errors), 0),
        "Paired-metric identity check failed.",
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = run_dir / "bootstrap_analysis" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    summary = []

    for block_length in BLOCK_LENGTHS:
        rng = np.random.default_rng(SEED + block_length)
        draws = np.empty((REPETITIONS, len(models), 2), dtype=float)

        for repetition in range(REPETITIONS):
            indices = circular_indices(groups, block_length, rng)

            if repetition == 0:
                require(
                    len(indices) == len(predictions),
                    "Bootstrap sample size changed.",
                )
                for year, expected in expected_counts.items():
                    require(
                        int((years[indices] == year).sum()) == expected,
                        f"Bootstrap changed the sample count for {year}.",
                    )

            draws[repetition] = metric_differences(errors[indices])

        require(
            np.isfinite(draws).all(),
            "Bootstrap produced nonfinite metric differences.",
        )

        np.savez_compressed(
            output_dir / f"draws_block_{block_length}.npz",
            differences=draws,
            models=np.asarray(models),
            metrics=np.asarray(["MAE", "RMSE"]),
        )

        for model_index, model in enumerate(models):
            for metric_index, metric in enumerate(("MAE", "RMSE")):
                lower, upper = np.quantile(
                    draws[:, model_index, metric_index],
                    [0.025, 0.975],
                )

                summary.append({
                    "Horizon": horizon,
                    "Block_Length": block_length,
                    "Model": model,
                    "Metric": metric,
                    "Samples": len(predictions),
                    "Difference_USD": float(
                        observed[model_index, metric_index]
                    ),
                    "CI_Lower_USD": float(lower),
                    "CI_Upper_USD": float(upper),
                    "Includes_Zero": bool(lower <= 0 <= upper),
                })

        logger.info(
            "Block length %s: %s paired repetitions completed",
            block_length,
            REPETITIONS,
        )

    table = pd.DataFrame(summary)
    table.to_csv(output_dir / "bootstrap_summary.csv", index=False)

    settings = {
        "source_run": str(run_dir),
        "horizon": horizon,
        "method": "paired circular block bootstrap within scoring year",
        "repetitions": REPETITIONS,
        "seed": SEED,
        "block_lengths": list(BLOCK_LENGTHS),
        "interval": "95% percentile",
        "difference": "model metric minus naive metric",
        "models": models,
        "limitations": [
            "Conditional on saved predictions; models are not retrained.",
            "Preserves dependence within blocks, not all time dependence.",
            "Assumes within-year resampling is a useful approximation.",
            "Does not adjust for prior model selection or multiple comparisons.",
            "Development analysis only; no final-test observations used.",
        ],
    }
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    print("\nMetric differences: model minus naive, in USD")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    logger.info("Paired sampling and metric checks PASS")
    logger.info("Saved bootstrap analysis: %s", output_dir)


if __name__ == "__main__":
    run_logged(main, logger)