import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import PROJECT_ROOT
from src.evaluation.bootstrap import (
    circular_indices,
    metric_differences,
)
from src.evaluation.freeze_final import read_json, require, sha256
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.bootstrap_final")

REPETITIONS = 5000
BLOCK_LENGTHS = (5, 10, 20)
SEED = 42

MODELS = {
    1: [
        "Training_mean_change",
        "LSTM_h1_w120",
        "Attention_LSTM_h1_w120",
        "GRU_h1_w60",
        "XGBoost_h1_w120",
        "ARIMA_h1",
    ],
    5: [
        "Training_mean_change",
        "LSTM_h5_w120",
    ],
}

EXPECTED_COUNTS = {
    1: {2023: 250, 2024: 252, 2025: 250},
    5: {2023: 246, 2024: 252, 2025: 250},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    results_dir = results_dir.resolve()

    completion = read_json(results_dir / "completion.json")
    require(
        completion["status"] == "completed",
        "Final evaluation did not complete.",
    )
    require(
        sha256(results_dir / "frozen_manifest.json")
        == completion["manifest_sha256"],
        "Result manifest fingerprint mismatch.",
    )

    pooled = pd.read_csv(results_dir / "pooled_metrics.csv")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = results_dir / "bootstrap_analysis" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    summary = []
    source_hashes = {}

    for horizon, models in MODELS.items():
        path = results_dir / f"h{horizon}_predictions.csv"
        source_hashes[path.name] = sha256(path)

        predictions = pd.read_csv(
            path,
            parse_dates=["Forecast_Date", "Target_Date"],
        )
        required = [
            "Forecast_Date", "Target_Date",
            "Actual_Close", "Naive", "Naive_Prediction",
            *models,
        ]
        require(
            all(column in predictions.columns for column in required),
            f"Horizon {horizon}: missing required columns.",
        )

        dates = predictions["Target_Date"]
        require(
            predictions[["Forecast_Date", "Target_Date"]]
            .notna().all().all()
            and dates.is_unique
            and dates.is_monotonic_increasing
            and (predictions["Forecast_Date"] < dates).all(),
            f"Horizon {horizon}: invalid forecast dates.",
        )

        years = dates.dt.year.to_numpy()
        counts = EXPECTED_COUNTS[horizon]
        require(
            set(years) == set(counts),
            "Unexpected test target years.",
        )

        groups = []
        for year, expected in counts.items():
            positions = np.flatnonzero(years == year)
            require(
                len(positions) == expected,
                f"Horizon {horizon}, year {year}: unexpected count.",
            )
            groups.append(positions)

        actual = predictions["Actual_Close"].to_numpy(dtype=float)
        forecasts = predictions[
            ["Naive", *models]
        ].to_numpy(dtype=float)

        require(
            np.isfinite(actual).all()
            and (actual > 0).all()
            and np.isfinite(forecasts).all(),
            "Invalid actual values or predictions.",
        )
        require(
            np.allclose(
                forecasts[:, 0],
                predictions["Naive_Prediction"].to_numpy(dtype=float),
            ),
            "Naive prediction columns differ.",
        )

        errors = forecasts - actual[:, None]
        observed = metric_differences(errors)

        # Verify these predictions reproduce the saved pooled metrics.
        for column, name in enumerate(["Naive", *models]):
            saved = pooled.loc[
                (pooled["Horizon"] == horizon)
                & (pooled["Model"] == name)
            ]
            require(len(saved) == 1, f"Missing pooled metric row: {name}")

            mae = float(np.mean(np.abs(errors[:, column])))
            rmse = float(np.sqrt(np.mean(errors[:, column] ** 2)))

            require(
                np.isclose(mae, saved.iloc[0]["MAE"], rtol=1e-7, atol=1e-8)
                and np.isclose(
                    rmse, saved.iloc[0]["RMSE"], rtol=1e-7, atol=1e-8
                ),
                f"{name}: saved predictions and metrics disagree.",
            )

        for block_length in BLOCK_LENGTHS:
            rng = np.random.default_rng(SEED + block_length)
            draws = np.empty((REPETITIONS, len(models), 2))

            for repetition in range(REPETITIONS):
                indices = circular_indices(groups, block_length, rng)

                if repetition == 0:
                    require(
                        len(indices) == len(predictions)
                        and all(
                            int((years[indices] == year).sum()) == count
                            for year, count in counts.items()
                        ),
                        "Bootstrap changed yearly sample counts.",
                    )

                # Identical sampled dates for naive and every model.
                draws[repetition] = metric_differences(errors[indices])

            require(
                np.isfinite(draws).all(),
                "Nonfinite bootstrap results.",
            )

            np.savez_compressed(
                output_dir / f"h{horizon}_block{block_length}_draws.npz",
                differences=draws,
                models=np.asarray(models),
                metrics=np.asarray(["MAE", "RMSE"]),
            )

            for model_index, name in enumerate(models):
                for metric_index, metric in enumerate(("MAE", "RMSE")):
                    lower, upper = np.quantile(
                        draws[:, model_index, metric_index],
                        [0.025, 0.975],
                    )

                    summary.append({
                        "Horizon": horizon,
                        "Block_Length": block_length,
                        "Model": name,
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
                "Horizon %s, block %s: %s paired repetitions completed",
                horizon, block_length, REPETITIONS,
            )

    table = pd.DataFrame(summary)
    table.to_csv(output_dir / "bootstrap_summary.csv", index=False)

    import json
    settings = {
        "results_directory": str(results_dir),
        "source_prediction_sha256": source_hashes,
        "manifest_sha256": completion["manifest_sha256"],
        "method": "paired circular block bootstrap within target year",
        "repetitions": REPETITIONS,
        "block_lengths": list(BLOCK_LENGTHS),
        "seed": SEED,
        "interval": "95% percentile, individual comparisons",
        "difference": "model metric minus naive metric",
        "models": MODELS,
        "limitations": [
            "Conditional on saved predictions; no retraining.",
            "No multiple-comparison adjustment.",
            "Within-year resampling is an approximation.",
            "Blocks preserve some, not all, temporal dependence.",
            "Year-boundary dependence is not preserved.",
            "Descriptive final-test analysis; not model selection.",
        ],
    }
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    for horizon in MODELS:
        print(f"\nHorizon {horizon}: model minus naive, USD")
        print(
            table.loc[table["Horizon"] == horizon].to_string(
                index=False, float_format="%.6f"
            )
        )

    logger.info("Saved-metric and paired-sampling checks PASS")
    logger.info("Saved final-test bootstrap: %s", output_dir)


if __name__ == "__main__":
    run_logged(main, logger)