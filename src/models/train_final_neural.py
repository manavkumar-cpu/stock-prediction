import argparse
import hashlib
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


logger = get_logger("models.train_final_neural")

FEATURES = [
    "Close", "Return_1", "RSI_14", "Volatility_20",
    "Relative_Volume_20", "Range_Ratio",
]

JOBS = [
    ("LSTM_h1_w120", "h1_w120", "lstm", 1, 120),
    ("Attention_LSTM_h1_w120", "h1_w120", "attention", 1, 120),
    ("GRU_h1_w60", "h1_w60", "gru", 1, 60),
    ("LSTM_h5_w120", "h5_w120", "lstm", 5, 120),
]


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, contents):
    path.write_text(
        json.dumps(contents, indent=2), encoding="utf-8"
    )


def load_partition(folder, split, window, target_range):
    with np.load(folder / f"{split}.npz", allow_pickle=False) as arrays:
        X = arrays["X"]
        y = arrays["y"]

    date_columns = [
        "Window_Start", "Forecast_Date",
        "Target_Start_Date", "Target_Date",
    ]
    metadata = pd.read_csv(
        folder / f"{split}_metadata.csv",
        parse_dates=date_columns,
    )

    n = len(metadata)
    require(
        n > 1
        and X.shape == (n, window, len(FEATURES))
        and y.shape == (n, 1),
        f"{split}: unexpected array shapes.",
    )
    require(
        np.isfinite(X).all() and np.isfinite(y).all(),
        f"{split}: nonfinite arrays.",
    )

    dates = metadata["Target_Date"]
    require(
        metadata[date_columns].notna().all().all()
        and dates.is_unique
        and dates.is_monotonic_increasing,
        f"{split}: invalid dates.",
    )
    require(
        (metadata["Window_Start"] <= metadata["Forecast_Date"]).all()
        and (
            metadata["Forecast_Date"] < metadata["Target_Start_Date"]
        ).all()
        and (metadata["Target_Start_Date"] <= dates).all(),
        f"{split}: invalid forecast alignment.",
    )

    if split == "train":
        valid = (
            (metadata["Target_Start_Date"] >= "2010-01-01")
            & (dates <= "2019-12-31")
        )
    else:
        valid = (
            (metadata["Target_Start_Date"] > "2019-12-31")
            & (dates <= "2022-12-31")
        )
    require(valid.all(), f"{split}: dates outside the permitted period.")

    prices = metadata[
        ["Actual_Close", "Naive_Prediction"]
    ].to_numpy(dtype=float)
    require(
        np.isfinite(prices).all() and (prices > 0).all(),
        f"{split}: invalid prices.",
    )
    require(
        np.allclose(X[:, -1, 0], 0),
        f"{split}: relative Close must end at zero.",
    )
    require(
        np.allclose(
            y.astype(float).ravel() * target_range + prices[:, 1],
            prices[:, 0],
            rtol=1e-6,
            atol=1e-5,
        ),
        f"{split}: target reconstruction failed.",
    )

    return X, y, metadata


def build_model(kind, window, settings):
    units = settings["lstm_units"]
    inputs = tf.keras.Input(
        shape=(window, len(FEATURES)), name="feature_history"
    )

    if kind == "attention":
        hidden = tf.keras.layers.LSTM(
            units, return_sequences=True, name="lstm"
        )(inputs)
        scores = tf.keras.layers.Dense(
            1, use_bias=False, name="attention_scores"
        )(hidden)
        weights = tf.keras.layers.Softmax(
            axis=1, name="attention_weights"
        )(scores)
        context = tf.keras.layers.Dot(
            axes=(1, 1), name="weighted_context"
        )([weights, hidden])
        encoded = tf.keras.layers.Flatten()(context)
    elif kind == "gru":
        encoded = tf.keras.layers.GRU(
            units, reset_after=True, name="gru"
        )(inputs)
    else:
        encoded = tf.keras.layers.LSTM(units, name="lstm")(inputs)

    outputs = tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="scaled_correction",
    )(encoded)

    model = tf.keras.Model(inputs, outputs, name=f"final_{kind}")
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

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_dir = data_dir.resolve()

    base = read_json(data_dir / "base_config_snapshot.json")
    settings = base["training"]
    seed = base["project"]["random_seed"]

    require(
        settings == {
            "lstm_units": 32,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "patience": 10,
        } and seed == 42,
        "Training settings differ from the agreed protocol.",
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = (
        PROJECT_ROOT / base["paths"]["artifacts"]
        / "final_neural" / stamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    save_json(run_dir / "run_info.json", {
        "prepared_data": str(data_dir),
        "tensorflow_version": tf.__version__,
        "seed": seed,
        "training": settings,
        "model_output": "scaled_dollar_correction",
        "checkpoint_selection": "minimum validation MSE",
        "refit_on_validation": False,
        "test_evaluated": False,
    })

    tf.config.experimental.enable_op_determinism()
    training_rows = []
    metric_rows = []
    selected_models = []

    for name, dataset, kind, horizon, window in JOBS:
        source = data_dir / dataset
        experiment = read_json(source / "experiment.json")
        preprocessing = read_json(source / "preprocessing.json")

        require(
            experiment["features"] == FEATURES
            and experiment["horizon"] == horizon
            and experiment["window_size"] == window
            and experiment["target"] == "future_close_minus_current_close",
            f"{name}: unexpected prepared dataset.",
        )
        require(
            pd.Timestamp(preprocessing["feature_fit_last_date"])
            <= pd.Timestamp("2019-12-31")
            and pd.Timestamp(preprocessing["target_fit_last_date"])
            <= pd.Timestamp("2019-12-31"),
            f"{name}: preprocessing extends beyond training.",
        )

        target_range = float(preprocessing["target_range"])
        require(
            np.isfinite(target_range) and target_range > 0,
            f"{name}: invalid target range.",
        )

        X_train, y_train, train_meta = load_partition(
            source, "train", window, target_range
        )
        X_val, y_val, val_meta = load_partition(
            source, "validation", window, target_range
        )

        folder = run_dir / name
        folder.mkdir()
        for filename in (
            "experiment.json", "preprocessing.json", "feature_scaler.pkl"
        ):
            shutil.copy2(source / filename, folder / filename)

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_model(kind, window, settings)

        require(
            np.allclose(model(X_train[:8], training=False).numpy(), 0),
            f"{name}: initial zero-correction check failed.",
        )

        checkpoint = folder / "best.keras"
        logger.info(
            "%s: train=%s, validation=%s, parameters=%s",
            name, X_train.shape, X_val.shape, model.count_params(),
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=settings["epochs"],
            batch_size=settings["batch_size"],
            shuffle=False,
            callbacks=[
                CheckFiniteLoss(),
                tf.keras.callbacks.ModelCheckpoint(
                    str(checkpoint),
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
        history_table.to_csv(folder / "history.csv")

        best_epoch = int(history_table["val_loss"].idxmin())
        best_loss = float(history_table.loc[best_epoch, "val_loss"])

        restored = tf.keras.models.load_model(checkpoint)
        checked = restored.evaluate(
            X_val, y_val,
            batch_size=settings["batch_size"],
            verbose=0,
            return_dict=True,
        )
        require(
            np.isclose(checked["loss"], best_loss, rtol=1e-5, atol=1e-8),
            f"{name}: checkpoint reload check failed.",
        )

        if kind == "attention":
            reader = tf.keras.Model(
                restored.inputs,
                restored.get_layer("attention_weights").output,
            )
            weights = reader(X_val[:16], training=False).numpy()
            require(
                np.isfinite(weights).all()
                and (weights >= 0).all()
                and np.allclose(weights.sum(axis=1), 1, atol=1e-6),
                "Attention normalization failed.",
            )

        correction = restored.predict(
            X_val, batch_size=settings["batch_size"], verbose=0
        )
        require(
            correction.shape == (len(val_meta), 1)
            and np.isfinite(correction).all(),
            f"{name}: invalid validation predictions.",
        )

        actual = val_meta["Actual_Close"].to_numpy(dtype=float)
        today = val_meta["Naive_Prediction"].to_numpy(dtype=float)
        drift = float((
            train_meta["Actual_Close"] - train_meta["Naive_Prediction"]
        ).mean())

        predictions = {
            "Naive": today,
            "Training_mean_change": today + drift,
            name: today + correction.astype(float).ravel() * target_range,
        }

        for model_name, predicted in predictions.items():
            val_meta[model_name] = predicted
            metric_rows.append({
                "Dataset": dataset,
                "Run_Model": name,
                "Model": model_name,
                "Horizon": horizon,
                "Samples": len(actual),
                **regression_metrics(actual, predicted),
                "Mean_Error": float(np.mean(predicted - actual)),
            })

        val_meta.to_csv(folder / "validation_predictions.csv", index=False)

        training_rows.append({
            "Model": name,
            "Dataset": dataset,
            "Parameters": model.count_params(),
            "Epochs_Completed": len(history_table),
            "Best_Epoch": best_epoch,
            "Validation_MSE_Scaled": best_loss,
        })

        selected_models.append({
            "model": name,
            "architecture": kind,
            "dataset": dataset,
            "horizon": horizon,
            "window": window,
            "features": FEATURES,
            "checkpoint": str(checkpoint.relative_to(run_dir)),
            "checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
            "best_epoch": best_epoch,
        })

        pd.DataFrame(training_rows).to_csv(
            run_dir / "training_summary.csv", index=False
        )
        pd.DataFrame(metric_rows).to_csv(
            run_dir / "validation_metrics.csv", index=False
        )
        logger.info("%s: checkpoint reload check PASS", name)

    save_json(run_dir / "selected_models.json", selected_models)

    print("\nSelected checkpoints:")
    print(pd.DataFrame(training_rows).to_string(index=False))
    print("\nValidation results:")
    print(pd.DataFrame(metric_rows).to_string(
        index=False, float_format="%.6f"
    ))
    logger.info("Saved final neural checkpoints: %s", run_dir)
    logger.info("Final-test data has not been evaluated")


if __name__ == "__main__":
    run_logged(main, logger)