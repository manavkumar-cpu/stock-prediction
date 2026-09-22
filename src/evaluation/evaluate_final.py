import argparse
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMAResults

from src.data.download import PROJECT_ROOT
from src.evaluation.freeze_final import read_json, require, sha256
from src.evaluation.metrics import regression_metrics
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.evaluate_final")

EXPECTED_MANIFEST_HASH = (
    "65f4e588c479dacbf8be41d1ae83b8d21"
    "e46745d79fc4169d025739fd2d39803"
)

EXPECTED_MODELS = {
    "LSTM_h1_w120",
    "Attention_LSTM_h1_w120",
    "GRU_h1_w60",
    "LSTM_h5_w120",
    "XGBoost_h1_w120",
    "ARIMA_h1",
}


def save_json(path, contents):
    import json
    path.write_text(
        json.dumps(contents, indent=2), encoding="utf-8"
    )


def transform_windows(values, positions, window, preprocessing, scaler):
    rows = (
        positions[:, None]
        + np.arange(-window + 1, 1)[None, :]
    )
    require((rows >= 0).all(), "Insufficient input history.")

    windows = values[rows]
    require(np.isfinite(windows).all(), "Incomplete input windows.")

    close_range = float(preprocessing["close_range"])
    require(
        np.isfinite(close_range) and close_range > 0,
        "Invalid saved Close range.",
    )

    relative_close = (
        windows[:, :, :1] - windows[:, -1:, :1]
    ) / close_range

    extras = scaler.transform(
        windows[:, :, 1:].reshape(-1, values.shape[1] - 1)
    ).reshape(len(positions), window, values.shape[1] - 1)

    X = np.concatenate([relative_close, extras], axis=2).astype(np.float32)
    require(np.isfinite(X).all(), "Nonfinite transformed inputs.")
    require(np.allclose(X[:, -1, 0], 0), "Close centering failed.")
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest_path = manifest_path.resolve()

    require(
        sha256(manifest_path) == EXPECTED_MANIFEST_HASH,
        "Manifest fingerprint differs from the reviewed frozen manifest.",
    )
    manifest = read_json(manifest_path)

    for relative, fingerprint in manifest["files"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Frozen file is missing: {relative}")
        require(
            sha256(path) == fingerprint["sha256"],
            f"Frozen file changed: {relative}",
        )

    require(
        len(manifest["models"]) == 6
        and {m["model"] for m in manifest["models"]} == EXPECTED_MODELS,
        "Unexpected frozen model registry.",
    )
    logger.info("Manifest and all frozen file fingerprints PASS")

    prepared = PROJECT_ROOT / manifest["prepared_data"]
    validation_end = pd.Timestamp(manifest["split"]["validation_end"])
    test_end = pd.Timestamp(manifest["split"]["test_end"])

    raw = pd.read_csv(
        PROJECT_ROOT / manifest["raw_data"],
        parse_dates=["Date"], index_col="Date",
    )
    features = pd.read_csv(
        PROJECT_ROOT / manifest["feature_data"],
        parse_dates=["Date"], index_col="Date",
    )

    for name, table in (("raw", raw), ("features", features)):
        require(
            table.index.notna().all()
            and table.index.is_unique
            and table.index.is_monotonic_increasing,
            f"{name}: invalid date index.",
        )

    require(raw.index.equals(features.index), "Raw/feature dates differ.")
    require(
        np.allclose(raw["Close"], features["Close"]),
        "Raw/feature Close values differ.",
    )

    features = features.loc[features.index <= test_end]
    close = raw.loc[features.index, "Close"]
    require(
        np.isfinite(close.to_numpy()).all() and (close > 0).all(),
        "Invalid Close values.",
    )

    datasets = {}
    expected_counts = {"h1_w120": 752, "h1_w60": 752, "h5_w120": 748}

    for dataset in expected_counts:
        folder = prepared / dataset
        experiment = read_json(folder / "experiment.json")
        preprocessing = read_json(folder / "preprocessing.json")
        columns = experiment["features"]
        horizon = int(experiment["horizon"])
        window = int(experiment["window_size"])

        require(
            pd.Timestamp(preprocessing["feature_fit_last_date"])
            <= pd.Timestamp("2019-12-31")
            and pd.Timestamp(preprocessing["target_fit_last_date"])
            <= pd.Timestamp("2019-12-31"),
            "Preprocessing extends beyond training.",
        )

        with (folder / "feature_scaler.pkl").open("rb") as file:
            scaler = pickle.load(file)

        values = features[columns].to_numpy(dtype=float)

        # Check reconstruction on validation before any model test inference.
        val_meta = pd.read_csv(
            folder / "validation_metadata.csv",
            parse_dates=["Forecast_Date"],
        )
        val_positions = features.index.get_indexer(val_meta["Forecast_Date"])
        require((val_positions >= 0).all(), "Validation input dates missing.")

        reconstructed = transform_windows(
            values, val_positions, window, preprocessing, scaler
        )
        with np.load(folder / "validation.npz", allow_pickle=False) as arrays:
            require(
                np.array_equal(reconstructed, arrays["X"]),
                f"{dataset}: validation input reconstruction differs.",
            )

        positions = np.arange(window - 1, len(features) - horizon)
        starts = features.index[positions + 1]
        targets = features.index[positions + horizon]

        # Entire future target span belongs to the continuous test partition.
        keep = (starts > validation_end) & (targets <= test_end)
        positions = positions[keep]

        require(
            len(positions) == expected_counts[dataset],
            f"{dataset}: unexpected test sample count.",
        )

        X = transform_windows(
            values, positions, window, preprocessing, scaler
        )
        metadata = pd.DataFrame({
            "Window_Start": features.index[positions - window + 1],
            "Forecast_Date": features.index[positions],
            "Target_Start_Date": features.index[positions + 1],
            "Target_Date": features.index[positions + horizon],
            "Actual_Close": close.to_numpy()[positions + horizon],
            "Naive_Prediction": close.to_numpy()[positions],
        })

        require(
            metadata["Target_Date"].is_unique
            and metadata["Target_Date"].is_monotonic_increasing,
            "Invalid test target dates.",
        )

        datasets[dataset] = {
            "X": X,
            "metadata": metadata,
            "preprocessing": preprocessing,
        }
        logger.info(
            "%s: validation reconstruction PASS; test inputs=%s",
            dataset, X.shape,
        )

    # One-session models must score exactly the same observations.
    for column in ("Forecast_Date", "Target_Date", "Actual_Close",
                   "Naive_Prediction"):
        require(
            np.array_equal(
                datasets["h1_w120"]["metadata"][column].to_numpy(),
                datasets["h1_w60"]["metadata"][column].to_numpy(),
            ),
            f"One-session alignment failed: {column}",
        )

    tables = {
        1: datasets["h1_w120"]["metadata"].copy(),
        5: datasets["h5_w120"]["metadata"].copy(),
    }
    model_names = {}

    for horizon, table in tables.items():
        table["Naive"] = table["Naive_Prediction"]
        drift_path = (
            manifest["baselines"]["training_mean_change"][f"h{horizon}"]
        )
        drift = float(read_json(PROJECT_ROOT / drift_path)["training_mean_change"])
        require(np.isfinite(drift), "Invalid training-mean-change baseline.")
        table["Training_mean_change"] = table["Naive"] + drift
        model_names[horizon] = ["Naive", "Training_mean_change"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = manifest_path.parent / f"test_results_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(manifest_path, output_dir / "frozen_manifest.json")
    shutil.copy2(Path(__file__), output_dir / "evaluate_final.py")
    save_json(output_dir / "execution.json", {
        "status": "started",
        "manifest_sha256": EXPECTED_MANIFEST_HASH,
        "evaluator_sha256": sha256(Path(__file__)),
        "tensorflow_version": tf.__version__,
        "xgboost_version": xgb.__version__,
        "refit": False,
    })

    for entry in manifest["models"]:
        name = entry["model"]
        horizon = int(entry["horizon"])
        table = tables[horizon]
        checkpoint = PROJECT_ROOT / entry["checkpoint"]

        if entry["output"] == "price_usd":
            state = ARIMAResults.load(str(checkpoint))
            original_params = np.asarray(state.params).copy()

            require(
                pd.Timestamp(entry["state_as_of"])
                == table["Forecast_Date"].iloc[0],
                "ARIMA saved state is not aligned with the first forecast.",
            )
            require(
                np.array_equal(
                    table["Forecast_Date"].iloc[1:].to_numpy(),
                    table["Target_Date"].iloc[:-1].to_numpy(),
                ),
                "ARIMA test observations are not consecutive.",
            )

            predictions = []
            for actual in table["Actual_Close"].to_numpy(dtype=float):
                prediction = float(np.asarray(state.forecast(steps=1))[0])
                require(np.isfinite(prediction), "Nonfinite ARIMA prediction.")
                predictions.append(prediction)
                state = state.extend(np.asarray([actual]))

            require(
                np.array_equal(original_params, np.asarray(state.params)),
                "ARIMA coefficients changed during test.",
            )
            predicted = np.asarray(predictions)

        else:
            dataset = datasets[entry["dataset"]]
            X = dataset["X"]
            today = table["Naive_Prediction"].to_numpy(dtype=float)

            if entry["output"] == "dollar_correction":
                model = xgb.XGBRegressor()
                model.load_model(checkpoint)
                correction = model.predict(
                    X.reshape(len(X), -1),
                    iteration_range=tuple(entry["prediction_iteration_range"]),
                ).astype(float)
                predicted = today + correction

            else:
                require(
                    entry["output"] == "scaled_dollar_correction",
                    "Unknown model output representation.",
                )
                tf.keras.backend.clear_session()
                model = tf.keras.models.load_model(checkpoint, compile=False)
                correction = model.predict(
                    X, batch_size=32, verbose=0
                )
                require(
                    correction.shape == (len(table), 1),
                    f"{name}: unexpected neural output shape.",
                )
                target_range = float(dataset["preprocessing"]["target_range"])
                predicted = (
                    today + correction.astype(float).ravel() * target_range
                )

        require(
            predicted.shape == (len(table),)
            and np.isfinite(predicted).all(),
            f"{name}: invalid test predictions.",
        )

        table[name] = predicted
        model_names[horizon].append(name)
        logger.info("%s: test predictions completed", name)

    metric_rows = []

    for horizon, table in tables.items():
        table.to_csv(output_dir / f"h{horizon}_predictions.csv", index=False)

        groups = [("Pooled", table)]
        groups.extend(
            (str(year), subset)
            for year, subset in table.groupby(table["Target_Date"].dt.year)
        )

        for period, subset in groups:
            actual = subset["Actual_Close"].to_numpy(dtype=float)
            naive_metrics = regression_metrics(
                actual, subset["Naive"].to_numpy(dtype=float)
            )

            for name in model_names[horizon]:
                predicted = subset[name].to_numpy(dtype=float)
                metrics = regression_metrics(actual, predicted)
                metric_rows.append({
                    "Horizon": horizon,
                    "Period": period,
                    "Model": name,
                    "Samples": len(subset),
                    **metrics,
                    "Mean_Error": float(np.mean(predicted - actual)),
                    "MAE_Improvement_Percent": (
                        100 * (1 - metrics["MAE"] / naive_metrics["MAE"])
                    ),
                    "RMSE_Improvement_Percent": (
                        100 * (1 - metrics["RMSE"] / naive_metrics["RMSE"])
                    ),
                })

        learned = model_names[horizon][2:]
        fig, axes = plt.subplots(
            len(learned), 2,
            figsize=(15, 3.2 * len(learned)),
            squeeze=False,
        )
        for row, name in enumerate(learned):
            axes[row, 0].plot(
                table["Target_Date"], table["Actual_Close"],
                color="black", linewidth=0.8, label="Actual",
            )
            axes[row, 0].plot(
                table["Target_Date"], table[name],
                linewidth=0.8, label=name,
            )
            axes[row, 0].set_ylabel("Close (USD, split-adjusted)")
            axes[row, 0].legend(fontsize=8)

            for label in ("Naive", name):
                axes[row, 1].plot(
                    table["Target_Date"],
                    table[label] - table["Actual_Close"],
                    linewidth=0.7, label=label,
                )
            axes[row, 1].axhline(0, color="black", linestyle="--")
            axes[row, 1].set_ylabel("Prediction − actual (USD)")
            axes[row, 1].legend(fontsize=8)

        fig.suptitle(f"AAPL final test — horizon {horizon} sessions")
        fig.tight_layout()
        fig.savefig(output_dir / f"h{horizon}_forecasts.png", dpi=160)
        plt.close(fig)

    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(output_dir / "all_metrics.csv", index=False)
    pooled = metrics_table.loc[metrics_table["Period"] == "Pooled"]
    pooled.to_csv(output_dir / "pooled_metrics.csv", index=False)

    save_json(output_dir / "completion.json", {
        "status": "completed",
        "manifest_sha256": EXPECTED_MANIFEST_HASH,
        "models_evaluated": sorted(EXPECTED_MODELS),
        "h1_samples": len(tables[1]),
        "h5_samples": len(tables[5]),
        "test_results_now_observed": True,
    })

    print("\nFinal-test pooled results:")
    print(pooled.to_string(index=False, float_format="%.6f"))
    logger.info("Saved final-test results: %s", output_dir)


if __name__ == "__main__":
    run_logged(main, logger)