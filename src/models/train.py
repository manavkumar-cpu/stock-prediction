import json
import pickle
import platform
import shutil
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.download import PROJECT_ROOT, load_config
from src.exceptions import ConfigurationError, DataValidationError
from src.logger import get_logger, run_logged
from src.models.lstm import build_lstm


logger = get_logger("models.train")


class CheckFiniteLoss(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name in ["loss", "val_loss"]:
            value = logs.get(name)

            if value is None or not np.isfinite(value):
                raise DataValidationError(
                    f"Invalid {name} at epoch {epoch + 1}: {value}"
                )


def load_split(sequence_dir, split_name, input_shape, y_scaler, config):
    with np.load(
        sequence_dir / f"{split_name}.npz",
        allow_pickle=False,
    ) as arrays:
        X = arrays["X"]
        y = arrays["y"]

    metadata = pd.read_csv(
        sequence_dir / f"{split_name}_metadata.csv",
        index_col="Sample",
        parse_dates=["Window_Start", "Forecast_Date", "Target_Date"],
    )

    if (
        X.ndim != 3
        or X.shape[1:] != input_shape
        or len(X) == 0
        or y.shape != (len(X), 1)
        or len(metadata) != len(X)
    ):
        raise DataValidationError(f"{split_name}: invalid array shapes.")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise DataValidationError(f"{split_name}: nonfinite arrays.")

    if not np.array_equal(metadata.index, np.arange(len(metadata))):
        raise DataValidationError(f"{split_name}: invalid sample order.")

    if not metadata["Split"].eq(split_name).all():
        raise DataValidationError(f"{split_name}: incorrect split labels.")

    dates = metadata["Target_Date"]

    if (
        dates.isna().any()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
        or not (metadata["Forecast_Date"] < dates).all()
    ):
        raise DataValidationError(f"{split_name}: invalid date alignment.")

    train_end = pd.Timestamp(config["split"]["train_end"])
    validation_end = pd.Timestamp(config["split"]["validation_end"])

    if split_name == "train":
        valid_dates = dates <= train_end
    else:
        valid_dates = (dates > train_end) & (dates <= validation_end)

    if not valid_dates.all():
        raise DataValidationError(f"{split_name}: dates outside its split.")

    # Confirm scaled labels match original-price metadata.
    recovered = y_scaler.inverse_transform(
        y.astype(np.float64)
    ).ravel()

    np.testing.assert_allclose(
        recovered,
        metadata["Target_Close"].to_numpy(),
        rtol=1e-6,
        atol=1e-5,
    )

    logger.info("%s: X=%s, y=%s", split_name, X.shape, y.shape)

    return X, y


def main() -> None:
    config = load_config()
    forecast = config["forecast"]
    training = config["training"]
    experiment = config["experiment"]["name"]

    if (
        forecast["feature_columns"] != ["Close"]
        or forecast["target_column"] != "Close"
        or forecast["horizon"] != 1
    ):
        raise ConfigurationError(
            "This first run expects Close-only, one-session forecasting."
        )

    for key in ["lstm_units", "batch_size", "epochs", "patience"]:
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationError(f"{key} must be a positive integer.")

    window_size = forecast["window_size"]
    feature_count = len(forecast["feature_columns"])
    seed = config["project"]["random_seed"]

    sequence_dir = (
        PROJECT_ROOT
        / config["data"]["processed_dir"]
        / experiment
        / f"sequences_w{window_size}"
    )
    artifact_dir = (
        PROJECT_ROOT / config["paths"]["artifacts"] / experiment
    )

    sequence_settings = json.loads(
        (sequence_dir / "sequence_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    scaling_settings = json.loads(
        (artifact_dir / "scaling_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    expected = {
        "experiment": experiment,
        "ticker": config["data"]["ticker"].strip().upper(),
        "feature_columns": forecast["feature_columns"],
        "target_column": forecast["target_column"],
        "horizon": forecast["horizon"],
        "split": config["split"],
    }

    for key, value in expected.items():
        if (
            sequence_settings[key] != value
            or scaling_settings[key] != value
        ):
            raise ConfigurationError(f"Upstream metadata mismatch: {key}")

    if sequence_settings["window_size"] != window_size:
        raise ConfigurationError("Sequence window size mismatch.")

    # Load the scaler generated by our own scaling script.
    with (artifact_dir / "y_scaler.pkl").open("rb") as file:
        y_scaler = pickle.load(file)

    input_shape = (window_size, feature_count)

    X_train, y_train = load_split(
        sequence_dir, "train", input_shape, y_scaler, config
    )
    X_val, y_val = load_split(
        sequence_dir, "validation", input_shape, y_scaler, config
    )

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    model = build_lstm(
        window_size=window_size,
        feature_count=feature_count,
        units=training["lstm_units"],
        learning_rate=training["learning_rate"],
    )

    run_id = datetime.now(timezone.utc).strftime(
        "lstm_%Y%m%dT%H%M%S_%fZ"
    )
    run_dir = artifact_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # Preserve settings and the matching scalers with this model.
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    for filename in [
        "x_scaler.pkl", "y_scaler.pkl", "scaling_metadata.json"
    ]:
        shutil.copy2(artifact_dir / filename, run_dir / filename)

    shutil.copy2(
        sequence_dir / "sequence_metadata.json",
        run_dir / "sequence_metadata.json",
    )

    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    (run_dir / "model_summary.txt").write_text(
        summary_text, encoding="utf-8"
    )

    best_path = run_dir / "best.keras"

    callbacks = [
        CheckFiniteLoss(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=training["patience"],
            min_delta=0,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(run_dir / "epoch_log.csv")
        ),
    ]

    logger.info("Run directory: %s", run_dir)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        shuffle=False,
        callbacks=callbacks,
        verbose=2,
    )

    history_table = pd.DataFrame(history.history)
    history_table.index = np.arange(1, len(history_table) + 1)
    history_table.index.name = "Epoch"
    history_table.to_csv(run_dir / "history.csv")

    best_epoch = int(history_table["val_loss"].idxmin())
    best_loss = float(history_table.loc[best_epoch, "val_loss"])

    # Verify that the saved checkpoint can be loaded.
    saved_model = tf.keras.models.load_model(best_path)
    restored_loss = float(
        saved_model.evaluate(
            X_val, y_val, verbose=0, return_dict=True
        )["loss"]
    )

    np.testing.assert_allclose(
        restored_loss, best_loss, rtol=1e-5, atol=1e-8
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history_table.index, history_table["loss"], label="Training")
    ax.plot(
        history_table.index,
        history_table["val_loss"],
        label="Validation",
    )
    ax.axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        label=f"Best epoch: {best_epoch}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE in scaled target units")
    ax.set_title("First LSTM — learning curves")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

    run_summary = {
        "run_id": run_id,
        "seed": seed,
        "epochs_completed": len(history_table),
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_loss,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "parameters": model.count_params(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "devices": [
            device.name for device in tf.config.list_physical_devices()
        ],
        "checkpoint": "best.keras",
    }

    (run_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nTraining summary:")
    print(json.dumps(run_summary, indent=2))

    logger.info("Saved checkpoint reload check: PASS")
    logger.info("Saved model, history, and learning curves: %s", run_dir)


if __name__ == "__main__":
    run_logged(main, logger)