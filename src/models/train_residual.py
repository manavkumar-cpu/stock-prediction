import argparse
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.download import PROJECT_ROOT
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged
from src.models.train import CheckFiniteLoss, load_split


logger = get_logger("models.train_residual")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_residual_lstm(config, x_scaler, y_scaler):
    window_size = config["forecast"]["window_size"]
    settings = config["training"]
    layers = tf.keras.layers

    inputs = layers.Input(
        shape=(window_size, 1),
        name="scaled_close_history",
    )

    # Keep only the final session's scaled Close.
    last_close = layers.Cropping1D(
        cropping=(window_size - 1, 0),
        name="last_session",
    )(inputs)
    last_close = layers.Flatten(name="last_close")(last_close)

    # Center the window on its final observed Close.
    repeated_close = layers.RepeatVector(
        window_size,
        name="repeat_last_close",
    )(last_close)

    relative_history = layers.Subtract(
        name="relative_history",
    )([inputs, repeated_close])

    hidden = layers.LSTM(
        settings["lstm_units"],
        name="lstm",
    )(relative_history)

    correction = layers.Dense(
        1,
        activation="linear",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="scaled_price_correction",
    )(hidden)

    # Input and target scalers have different fitted ranges.
    # Convert today's Close from input units to target units.
    factor = float(y_scaler.scale_[0] / x_scaler.scale_[0])
    offset = float(
        y_scaler.min_[0] - factor * x_scaler.min_[0]
    )

    baseline = layers.Rescaling(
        scale=factor,
        offset=offset,
        name="today_close_in_target_units",
    )(last_close)

    prediction = layers.Add(
        name="predicted_scaled_close",
    )([baseline, correction])

    model = tf.keras.Model(
        inputs=inputs,
        outputs=prediction,
        name="residual_lstm",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=settings["learning_rate"]
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    args = parser.parse_args()

    source_run = Path(args.source_run)
    if not source_run.is_absolute():
        source_run = PROJECT_ROOT / source_run

    config = read_json(source_run / "config.json")
    forecast = config["forecast"]

    if (
        forecast["feature_columns"] != ["Close"]
        or forecast["target_column"] != "Close"
        or forecast["horizon"] != 1
        or forecast["window_size"] < 2
    ):
        raise DataValidationError(
            "This experiment requires Close-only, one-session forecasts "
            "and a window of at least two sessions."
        )

    config["model_variant"] = "relative_history_residual_lstm"

    sequence_dir = (
        PROJECT_ROOT
        / config["data"]["processed_dir"]
        / config["experiment"]["name"]
        / f"sequences_w{forecast['window_size']}"
    )

    if read_json(sequence_dir / "sequence_metadata.json") != read_json(
        source_run / "sequence_metadata.json"
    ):
        raise DataValidationError(
            "Sequence metadata differs from the original training run."
        )

    with (source_run / "x_scaler.pkl").open("rb") as file:
        x_scaler = pickle.load(file)

    with (source_run / "y_scaler.pkl").open("rb") as file:
        y_scaler = pickle.load(file)

    input_shape = (forecast["window_size"], 1)

    X_train, y_train = load_split(
        sequence_dir, "train", input_shape, y_scaler, config
    )
    X_validation, y_validation = load_split(
        sequence_dir, "validation", input_shape, y_scaler, config
    )

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["project"]["random_seed"])
    tf.config.experimental.enable_op_determinism()

    model = build_residual_lstm(config, x_scaler, y_scaler)

    # A zero correction must reproduce today's Close.
    initial_scaled = model.predict(X_validation, verbose=0)

    initial_prices = y_scaler.inverse_transform(
        initial_scaled.astype(np.float64)
    )
    today_prices = x_scaler.inverse_transform(
        X_validation[:, -1, :].astype(np.float64)
    )

    if not np.allclose(
        initial_prices, today_prices, rtol=1e-6, atol=1e-4
    ):
        raise DataValidationError(
            "Zero-correction model does not reproduce the naive baseline."
        )

    logger.info("Initial zero-correction baseline check: PASS")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = (
        PROJECT_ROOT
        / config["paths"]["artifacts"]
        / "close_only_delta_h1"
        / "runs"
        / f"residual_lstm_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    for filename in [
        "x_scaler.pkl",
        "y_scaler.pkl",
        "scaling_metadata.json",
        "sequence_metadata.json",
    ]:
        shutil.copy2(source_run / filename, run_dir / filename)

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    (run_dir / "model_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    model.summary()

    settings = config["training"]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        shuffle=False,
        callbacks=[
            CheckFiniteLoss(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / "best.keras"),
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
            tf.keras.callbacks.CSVLogger(
                str(run_dir / "epoch_log.csv")
            ),
        ],
        verbose=2,
    )

    history_table = pd.DataFrame(history.history)
    history_table.index = np.arange(1, len(history_table) + 1)
    history_table.index.name = "Epoch"
    history_table.to_csv(run_dir / "history.csv")

    best_epoch = int(history_table["val_loss"].idxmin())
    best_loss = float(history_table.loc[best_epoch, "val_loss"])

    reloaded = tf.keras.models.load_model(run_dir / "best.keras")
    checked = reloaded.evaluate(
        X_validation,
        y_validation,
        batch_size=settings["batch_size"],
        verbose=0,
        return_dict=True,
    )

    if not np.isclose(
        checked["loss"], best_loss, rtol=1e-5, atol=1e-8
    ):
        raise DataValidationError(
            "Reloaded checkpoint loss differs from recorded best loss."
        )

    report = {
        "run_id": run_dir.name,
        "model_variant": config["model_variant"],
        "source_run": str(source_run),
        "epochs_completed": len(history_table),
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_loss,
        "initial_naive_validation_mse_scaled": float(
            np.mean(
                (
                    initial_scaled.astype(np.float64)
                    - y_validation.astype(np.float64)
                ) ** 2
            )
        ),
    }

    (run_dir / "run_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nRun summary:")
    print(json.dumps(report, indent=2))

    logger.info("Saved checkpoint reload check: PASS")
    logger.info("Saved residual experiment: %s", run_dir)

    relative_run = run_dir.relative_to(PROJECT_ROOT)
    print("\nEvaluate this run with:")
    print(
        "python -m src.evaluation.lstm "
        f"--run-dir {relative_run}"
    )


if __name__ == "__main__":
    run_logged(main, logger)

    