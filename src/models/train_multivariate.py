import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.download import PROJECT_ROOT
from src.evaluation.metrics import regression_metrics
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged
from src.models.train import CheckFiniteLoss


logger = get_logger("models.train_multivariate")


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_partition(folder, split, window, feature_count, target_range, year):
    with np.load(folder / f"{split}.npz", allow_pickle=False) as arrays:
        X = arrays["X"]
        y = arrays["y"]

    metadata = pd.read_csv(
        folder / f"{split}_metadata.csv",
        parse_dates=["Window_Start", "Forecast_Date", "Target_Date"],
    )

    n = len(metadata)
    require(n > 1, f"{split}: insufficient samples.")
    require(
        X.shape == (n, window, feature_count) and y.shape == (n, 1),
        f"{split}: unexpected array shapes.",
    )
    require(
        np.isfinite(X).all() and np.isfinite(y).all(),
        f"{split}: arrays contain nonfinite values.",
    )

    dates = metadata["Target_Date"]
    require(
        metadata[
            ["Window_Start", "Forecast_Date", "Target_Date"]
        ].notna().all().all()
        and dates.is_unique
        and dates.is_monotonic_increasing,
        f"{split}: invalid or unordered dates.",
    )
    require(
        (metadata["Window_Start"] <= metadata["Forecast_Date"]).all()
        and (metadata["Forecast_Date"] < dates).all(),
        f"{split}: invalid window or forecast alignment.",
    )

    if split == "train":
        valid_period = dates.dt.year <= year - 2
    elif split == "selection":
        valid_period = dates.dt.year == year - 1
    else:
        valid_period = dates.dt.year == year

    require(valid_period.all(), f"{split}: dates outside expected partition.")

    prices = metadata[
        ["Actual_Close", "Naive_Prediction"]
    ].to_numpy(dtype=float)

    require(
        np.isfinite(prices).all() and (prices > 0).all(),
        f"{split}: invalid metadata prices.",
    )
    require(
        np.allclose(X[:, -1, 0], 0),
        f"{split}: centered Close does not end at zero.",
    )

    restored = (
        y.astype(np.float64).ravel() * target_range
        + metadata["Naive_Prediction"].to_numpy()
    )

    require(
        np.allclose(
            restored,
            metadata["Actual_Close"].to_numpy(),
            rtol=1e-6,
            atol=1e-5,
        ),
        f"{split}: target reconstruction failed.",
    )

    return X, y, metadata


def build_correction_model(window, feature_count, settings):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, feature_count)),
        tf.keras.layers.LSTM(settings["lstm_units"]),
        tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer="zeros",
            bias_initializer="zeros",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=settings["learning_rate"]
        ),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    experiment = read_json(data_dir / "experiment.json")
    config = experiment["config"]
    settings = config["training"]
    features = experiment["features"]
    window = experiment["window_size"]

    require(
        features == [
            "Close", "Return_1", "RSI_14", "Volatility_20",
            "Relative_Volume_20", "Range_Ratio",
        ],
        "Unexpected feature set or feature order.",
    )
    require(
        experiment["scoring_years"] == [2020, 2021, 2022],
        "Expected development scoring years 2020–2022.",
    )
    require(
        experiment["target"] == "next_close_minus_current_close",
        "Unexpected target representation.",
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = (
        PROJECT_ROOT
        / config["paths"]["artifacts"]
        / "multivariate_walk_forward"
        / stamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(data_dir / "experiment.json", run_dir / "experiment.json")
    (run_dir / "run_info.json").write_text(
        json.dumps({
            "prepared_data": str(data_dir.resolve()),
            "tensorflow_version": tf.__version__,
            "seed": config["project"]["random_seed"],
            "model_output": "scaled_dollar_correction",
        }, indent=2),
        encoding="utf-8",
    )

    tf.config.experimental.enable_op_determinism()

    metric_rows = []
    training_rows = []
    prediction_tables = []

    variants = {
        "Close_only_LSTM": 1,
        "Six_feature_LSTM": len(features),
    }

    for year in experiment["scoring_years"]:
        source_fold = data_dir / str(year)
        fold_dir = run_dir / str(year)
        fold_dir.mkdir()

        preprocessing = read_json(source_fold / "preprocessing.json")
        target_range = float(preprocessing["target_range"])

        require(
            np.isfinite(target_range) and target_range > 0,
            "Invalid target range.",
        )
        require(
            pd.Timestamp(preprocessing["feature_fit_last_date"]).year
            <= year - 2,
            "Feature preprocessing extends beyond the training period.",
        )

        for filename in ["preprocessing.json", "feature_scaler.pkl"]:
            shutil.copy2(source_fold / filename, fold_dir / filename)

        X_train, y_train, train_meta = load_partition(
            source_fold, "train", window, len(features), target_range, year
        )
        X_select, y_select, _ = load_partition(
            source_fold, "selection", window, len(features), target_range, year
        )

        # Train and select both checkpoints before loading scoring arrays.
        checkpoints = {}

        for name, count in variants.items():
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(
                config["project"]["random_seed"]
            )

            model = build_correction_model(window, count, settings)
            checkpoint = fold_dir / f"{name}.keras"

            logger.info(
                "%s | %s | features=%s | parameters=%s",
                year, name, count, model.count_params(),
            )

            history = model.fit(
                X_train[:, :, :count],
                y_train,
                validation_data=(X_select[:, :, :count], y_select),
                epochs=settings["epochs"],
                batch_size=settings["batch_size"],
                shuffle=False,
                callbacks=[
                    CheckFiniteLoss(),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=str(checkpoint),
                        monitor="val_loss",
                        mode="min",
                        save_best_only=True,
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        mode="min",
                        patience=settings["patience"],
                        min_delta=0.0,
                        restore_best_weights=True,
                    ),
                ],
                verbose=2,
            )

            history_table = pd.DataFrame(history.history)
            history_table.index = np.arange(1, len(history_table) + 1)
            history_table.index.name = "Epoch"
            history_table.to_csv(fold_dir / f"{name}_history.csv")

            best_epoch = int(history_table["val_loss"].idxmin())
            best_loss = float(history_table.loc[best_epoch, "val_loss"])

            restored = tf.keras.models.load_model(checkpoint)
            checked = restored.evaluate(
                X_select[:, :, :count],
                y_select,
                batch_size=settings["batch_size"],
                verbose=0,
                return_dict=True,
            )

            require(
                np.isclose(
                    checked["loss"], best_loss, rtol=1e-5, atol=1e-8
                ),
                f"{name}: checkpoint reload check failed.",
            )

            training_rows.append({
                "Score_Year": year,
                "Model": name,
                "Parameters": model.count_params(),
                "Epochs_Completed": len(history_table),
                "Best_Epoch": best_epoch,
                "Selection_MSE_Scaled": best_loss,
            })
            checkpoints[name] = checkpoint

            logger.info(
                "%s | %s | best epoch=%s | reload check PASS",
                year, name, best_epoch,
            )

        X_score, _, score_meta = load_partition(
            source_fold, "score", window, len(features), target_range, year
        )

        actual = score_meta["Actual_Close"].to_numpy(dtype=float)
        today = score_meta["Naive_Prediction"].to_numpy(dtype=float)

        drift = float((
            train_meta["Actual_Close"] - train_meta["Naive_Prediction"]
        ).mean())

        predictions = {
            "Naive": today,
            "Training_mean_change": today + drift,
        }

        for name, count in variants.items():
            restored = tf.keras.models.load_model(
                checkpoints[name], compile=False
            )
            scaled_correction = restored.predict(
                X_score[:, :, :count],
                batch_size=settings["batch_size"],
                verbose=0,
            )

            require(
                scaled_correction.shape == (len(actual), 1)
                and np.isfinite(scaled_correction).all(),
                f"{name}: invalid scoring predictions.",
            )

            predictions[name] = (
                today
                + scaled_correction.astype(np.float64).ravel()
                * target_range
            )

        for name, predicted in predictions.items():
            metric_rows.append({
                "Score_Year": year,
                "Model": name,
                "Samples": len(actual),
                **regression_metrics(actual, predicted),
                "Mean_Error": float(np.mean(predicted - actual)),
            })
            score_meta[name] = predicted

        score_meta.to_csv(fold_dir / "predictions.csv", index=False)
        prediction_tables.append(score_meta)

        pd.DataFrame(metric_rows).to_csv(
            run_dir / "fold_metrics.csv", index=False
        )
        pd.DataFrame(training_rows).to_csv(
            run_dir / "training_summary.csv", index=False
        )

    combined = pd.concat(prediction_tables, ignore_index=True)

    require(
        combined["Target_Date"].is_unique,
        "Scoring dates overlap across folds.",
    )

    combined.to_csv(run_dir / "all_predictions.csv", index=False)

    actual = combined["Actual_Close"].to_numpy(dtype=float)
    pooled_rows = []

    for name in ["Naive", "Training_mean_change", *variants]:
        predicted = combined[name].to_numpy(dtype=float)
        pooled_rows.append({
            "Model": name,
            "Samples": len(actual),
            **regression_metrics(actual, predicted),
            "Mean_Error": float(np.mean(predicted - actual)),
        })

    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(run_dir / "pooled_metrics.csv", index=False)

    print("\nTraining summary:")
    print(pd.DataFrame(training_rows).to_string(index=False))

    print("\nResults by scoring year:")
    print(
        pd.DataFrame(metric_rows).to_string(
            index=False, float_format="%.6f"
        )
    )

    print("\nPooled development results:")
    print(pooled.to_string(index=False, float_format="%.6f"))

    logger.info("Saved multivariate comparison: %s", run_dir)


if __name__ == "__main__":
    run_logged(main, logger)