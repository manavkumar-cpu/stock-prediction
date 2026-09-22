import json
import pickle
from src.logger import get_logger, run_logged
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from src.exceptions import ConfigurationError
from src.data.download import PROJECT_ROOT, load_config
from src.data.prepare import create_target_table
from src.features.market import check_daily_index
logger = get_logger("data.scale")

def main() -> None:
    config = load_config()
    settings = config["data"]
    forecast = config["forecast"]

    if forecast["horizon"] != 1:
        raise ConfigurationError(
            "This implementation expects horizon: 1."
        )

    ticker = settings["ticker"].strip().upper()
    start = settings["start_date"]
    end = settings["end_date_exclusive"]

    experiment = config["experiment"]["name"]
    columns = forecast["feature_columns"]

    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Input feature names must be nonempty and unique.")

    processed_dir = PROJECT_ROOT / settings["processed_dir"]

    features = pd.read_csv(
        processed_dir / f"{ticker}_{start}_{end}_features_market.csv",
        index_col="Date",
        parse_dates=["Date"],
    )
    check_daily_index(features, "Features")

    targets = pd.read_csv(
        processed_dir / f"{ticker}_{start}_{end}_targets_h1.csv",
        index_col="Forecast_Date",
        parse_dates=["Forecast_Date", "Target_Date"],
    )

    # Verify that saved targets still match the source and configuration.
    expected_targets = create_target_table(
        features,
        forecast["target_column"],
        config["split"],
    )

    pd.testing.assert_frame_equal(
        targets,
        expected_targets,
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    missing = [column for column in columns if column not in features]
    if missing:
        raise ValueError(f"Missing selected features: {missing}")

    inputs = features[columns].astype(float)

    if np.isinf(inputs.to_numpy()).any():
        raise ValueError("Input features contain infinite values.")

    train_targets = targets.loc[targets["Split"] == "train"]

    if train_targets.empty:
        raise ValueError("No training targets available.")

    last_train_forecast = train_targets.index.max()

    # Include historical rows needed by training windows.
    input_fit = inputs.loc[:last_train_forecast].copy()
    target_fit = train_targets[["Target_Close"]].copy()

    if input_fit.notna().sum().eq(0).any():
        raise ValueError("A selected feature has no training observations.")

    if not np.isfinite(target_fit.to_numpy()).all():
        raise ValueError("Training targets must be finite.")

    x_scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
    y_scaler = MinMaxScaler(feature_range=(0, 1), clip=False)

    x_scaler.fit(input_fit)
    y_scaler.fit(target_fit)

    scaled_inputs = pd.DataFrame(
        x_scaler.transform(inputs),
        index=inputs.index,
        columns=columns,
    )

    scaled_targets = targets.copy()
    scaled_targets["Target_Scaled"] = y_scaler.transform(
        targets[["Target_Close"]]
    ).ravel()

    # Scaling should preserve initial feature NaNs and all dates.
    if not scaled_inputs.isna().equals(inputs.isna()):
        raise ValueError("Scaling changed the missing-value pattern.")

    if not scaled_inputs.index.equals(features.index):
        raise ValueError("Scaling changed the feature dates.")

    if np.isinf(scaled_inputs.to_numpy()).any():
        raise ValueError("Scaled inputs contain infinite values.")

    if not np.isfinite(scaled_targets["Target_Scaled"]).all():
        raise ValueError("Scaled targets contain invalid values.")

    # Confirm that the transformations can be reversed.
    recovered_inputs = x_scaler.inverse_transform(
        scaled_inputs.loc[input_fit.index]
    )
    np.testing.assert_allclose(
        recovered_inputs,
        input_fit.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )

    recovered_targets = y_scaler.inverse_transform(
        y_scaler.transform(target_fit)
    )
    np.testing.assert_allclose(
        recovered_targets,
        target_fit.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )

    output_dir = processed_dir / experiment
    artifact_dir = PROJECT_ROOT / config["paths"]["artifacts"] / experiment

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scaled_inputs.to_csv(
        output_dir / "features_scaled.csv",
        index_label="Date",
    )
    scaled_targets.to_csv(
        output_dir / "targets_scaled.csv",
        index_label="Forecast_Date",
    )

    with (artifact_dir / "x_scaler.pkl").open("wb") as file:
        pickle.dump(x_scaler, file)

    with (artifact_dir / "y_scaler.pkl").open("wb") as file:
        pickle.dump(y_scaler, file)

    metadata = {
        "experiment": experiment,
        "ticker": ticker,
        "feature_columns": columns,
        "target_column": forecast["target_column"],
        "horizon": forecast["horizon"],
        "split": config["split"],
        "scaler": "MinMaxScaler",
        "feature_range": [0, 1],
        "clip": False,
        "sklearn_version": sklearn.__version__,
        "input_fit_start": str(input_fit.index.min().date()),
        "input_fit_end": str(input_fit.index.max().date()),
        "input_fit_rows": len(input_fit),
        "input_nonmissing_counts": {
            column: int(input_fit[column].notna().sum())
            for column in columns
        },
        "input_min": x_scaler.data_min_.tolist(),
        "input_max": x_scaler.data_max_.tolist(),
        "target_fit_start": str(
            train_targets["Target_Date"].min().date()
        ),
        "target_fit_end": str(
            train_targets["Target_Date"].max().date()
        ),
        "target_fit_rows": len(target_fit),
        "target_min": y_scaler.data_min_.tolist(),
        "target_max": y_scaler.data_max_.tolist(),
    }

    (artifact_dir / "scaling_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Experiment:", experiment)
    print("Selected input features:", columns)

    print(
        "\nInput scaler fit dates:",
        metadata["input_fit_start"],
        "to",
        metadata["input_fit_end"],
    )
    print("Input scaler fit rows:", len(input_fit))

    print(
        "\nTarget scaler fit dates:",
        metadata["target_fit_start"],
        "to",
        metadata["target_fit_end"],
    )
    print("Target scaler fit rows:", len(target_fit))

    print("\nInput training minimum:", x_scaler.data_min_)
    print("Input training maximum:", x_scaler.data_max_)
    print("Target training minimum:", y_scaler.data_min_)
    print("Target training maximum:", y_scaler.data_max_)

    print("\nScaled feature shape:", scaled_inputs.shape)
    print("Scaled target rows:", len(scaled_targets))
    logger.info("Input and target inverse checks: PASS")
    logger.info("Missing-value pattern preserved")
    logger.info("Saved scaled tables: %s", output_dir)
    logger.info("Saved scalers and metadata: %s", artifact_dir)


if __name__ == "__main__":
    run_logged(main, logger)