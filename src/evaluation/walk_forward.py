import copy
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from src.data.download import PROJECT_ROOT, load_config
from src.evaluation.metrics import regression_metrics
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged
from src.models.lstm import build_lstm
from src.models.train import CheckFiniteLoss
from src.models.train_residual import build_residual_lstm


logger = get_logger("evaluation.walk_forward")


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def main() -> None:
    config = load_config()
    forecast = config["forecast"]
    settings = config["training"]
    data_settings = config["data"]

    require(
        forecast["feature_columns"] == ["Close"]
        and forecast["target_column"] == "Close"
        and forecast["horizon"] == 1,
        "This walk-forward experiment supports Close-only, horizon 1.",
    )

    window = forecast["window_size"]
    require(window >= 2, "Window must contain at least two sessions.")

    ticker = data_settings["ticker"].strip().upper()
    start = data_settings["start_date"]
    end = data_settings["end_date_exclusive"]

    raw_path = (
        PROJECT_ROOT
        / data_settings["raw_dir"]
        / f"{ticker}_{start}_{end}_raw.csv"
    )

    raw = pd.read_csv(
        raw_path, parse_dates=["Date"], index_col="Date"
    )

    require(
        raw.index.notna().all()
        and raw.index.is_unique
        and raw.index.is_monotonic_increasing,
        "Raw dates must be valid, unique, and chronological.",
    )

    # Exclude final-test observations before building examples.
    development = raw.loc[
        (raw.index >= pd.Timestamp(start))
        & (raw.index < pd.Timestamp("2023-01-01")),
        ["Close"],
    ].copy()

    prices = development["Close"].to_numpy(dtype=float)

    require(
        len(prices) > window
        and np.isfinite(prices).all()
        and (prices > 0).all(),
        "Development Close data is insufficient or invalid.",
    )

    # Each position is the final observed session in an input window.
    positions = np.arange(window - 1, len(prices) - 1)

    X_raw = np.stack([
        prices[position - window + 1:position + 1]
        for position in positions
    ])[:, :, None]

    y_raw = prices[positions + 1].reshape(-1, 1)

    metadata = pd.DataFrame({
        "Window_Start": development.index[positions - window + 1],
        "Forecast_Date": development.index[positions],
        "Target_Date": development.index[positions + 1],
        "Actual_Close": y_raw.ravel(),
        "Naive_Prediction": prices[positions],
    })

    target_year = metadata["Target_Date"].dt.year.to_numpy()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (
        PROJECT_ROOT
        / config["paths"]["artifacts"]
        / "walk_forward"
        / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    tf.config.experimental.enable_op_determinism()

    metric_rows = []
    prediction_tables = []

    for score_year in [2020, 2021, 2022]:
        selection_year = score_year - 1
        fit_end_year = score_year - 2

        fit_mask = target_year <= fit_end_year
        selection_mask = target_year == selection_year
        score_mask = target_year == score_year

        require(
            fit_mask.any() and selection_mask.any() and score_mask.any(),
            f"Fold {score_year} contains an empty partition.",
        )

        fold_dir = output_dir / str(score_year)
        fold_dir.mkdir()

        # Fit x scaler on observed history through the last fit forecast.
        last_fit_position = positions[fit_mask][-1]

        x_scaler = MinMaxScaler(clip=False)
        y_scaler = MinMaxScaler(clip=False)

        x_scaler.fit(prices[:last_fit_position + 1].reshape(-1, 1))
        y_scaler.fit(y_raw[fit_mask])

        X = x_scaler.transform(
            X_raw.reshape(-1, 1)
        ).reshape(X_raw.shape).astype(np.float32)

        y = y_scaler.transform(y_raw).astype(np.float32)

        for name, scaler in [
            ("x_scaler.pkl", x_scaler),
            ("y_scaler.pkl", y_scaler),
        ]:
            with (fold_dir / name).open("wb") as file:
                pickle.dump(scaler, file)

        fold_info = {
            "fit_target_end": str(
                metadata.loc[fit_mask, "Target_Date"].max().date()
            ),
            "checkpoint_selection_year": selection_year,
            "scoring_year": score_year,
            "fit_samples": int(fit_mask.sum()),
            "selection_samples": int(selection_mask.sum()),
            "scoring_samples": int(score_mask.sum()),
            "x_scaler_last_fit_date": str(
                development.index[last_fit_position].date()
            ),
            "refit_after_checkpoint_selection": False,
        }

        (fold_dir / "fold.json").write_text(
            json.dumps(fold_info, indent=2), encoding="utf-8"
        )

        logger.info(
            "Scoring year %s: fit=%s, selection=%s, score=%s",
            score_year,
            fit_mask.sum(),
            selection_mask.sum(),
            score_mask.sum(),
        )

        scored = metadata.loc[score_mask].copy()
        actual = scored["Actual_Close"].to_numpy()
        today = scored["Naive_Prediction"].to_numpy()

        drift = float(np.mean(
            y_raw[fit_mask, 0]
            - metadata.loc[fit_mask, "Naive_Prediction"].to_numpy()
        ))

        predictions = {
            "Naive": today,
            "Training_mean_change": today + drift,
        }

        for model_name in ["Direct_LSTM", "Residual_LSTM"]:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(
                config["project"]["random_seed"]
            )

            if model_name == "Direct_LSTM":
                model = build_lstm(
                    window_size=window,
                    feature_count=1,
                    units=settings["lstm_units"],
                    learning_rate=settings["learning_rate"],
                )
            else:
                model = build_residual_lstm(
                    copy.deepcopy(config), x_scaler, y_scaler
                )

            logger.info(
                "Training %s for scoring year %s",
                model_name, score_year,
            )

            history = model.fit(
                X[fit_mask],
                y[fit_mask],
                validation_data=(
                    X[selection_mask], y[selection_mask]
                ),
                epochs=settings["epochs"],
                batch_size=settings["batch_size"],
                shuffle=False,
                callbacks=[
                    CheckFiniteLoss(),
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        mode="min",
                        patience=settings["patience"],
                        min_delta=0.0,
                        restore_best_weights=True,
                    ),
                ],
                verbose=0,
            )

            history_table = pd.DataFrame(history.history)
            history_table.index = np.arange(1, len(history_table) + 1)
            history_table.index.name = "Epoch"
            history_table.to_csv(
                fold_dir / f"{model_name}_history.csv"
            )

            best_epoch = int(history_table["val_loss"].idxmin())
            model.save(fold_dir / f"{model_name}.keras")

            predicted_scaled = model.predict(
                X[score_mask],
                batch_size=settings["batch_size"],
                verbose=0,
            )

            require(
                predicted_scaled.shape == (len(scored), 1)
                and np.isfinite(predicted_scaled).all(),
                f"Invalid {model_name} predictions.",
            )

            predictions[model_name] = y_scaler.inverse_transform(
                predicted_scaled.astype(np.float64)
            ).ravel()

            logger.info(
                "%s: best selection epoch=%s",
                model_name, best_epoch,
            )

        for model_name, predicted in predictions.items():
            scores = regression_metrics(actual, predicted)

            metric_rows.append({
                "Score_Year": score_year,
                "Model": model_name,
                "Samples": len(actual),
                **scores,
                "Mean_Error": float(np.mean(predicted - actual)),
            })

            scored[model_name] = predicted

        scored.to_csv(fold_dir / "predictions.csv", index=False)
        prediction_tables.append(scored)

        # Save progress after each completed fold.
        pd.DataFrame(metric_rows).to_csv(
            output_dir / "fold_metrics.csv", index=False
        )

    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    all_predictions.to_csv(
        output_dir / "all_predictions.csv", index=False
    )

    # Compute pooled metrics from individual predictions,
    # rather than averaging yearly RMSE values.
    pooled_rows = []
    actual = all_predictions["Actual_Close"].to_numpy()

    for model_name in [
        "Naive", "Training_mean_change", "Direct_LSTM", "Residual_LSTM"
    ]:
        predicted = all_predictions[model_name].to_numpy()
        pooled_rows.append({
            "Model": model_name,
            "Samples": len(actual),
            **regression_metrics(actual, predicted),
            "Mean_Error": float(np.mean(predicted - actual)),
        })

    fold_metrics = pd.DataFrame(metric_rows)
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(output_dir / "pooled_metrics.csv", index=False)

    print("\nResults by scoring year:")
    print(fold_metrics.to_string(index=False, float_format="%.6f"))

    print("\nPooled development results:")
    print(pooled.to_string(index=False, float_format="%.6f"))

    logger.info("Saved walk-forward experiment: %s", output_dir)


if __name__ == "__main__":
    run_logged(main, logger)