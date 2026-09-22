import copy
import hashlib
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.download import PROJECT_ROOT, load_config
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("data.prepare_final_development")

FEATURES = [
    "Close",
    "Return_1",
    "RSI_14",
    "Volatility_20",
    "Relative_Volume_20",
    "Range_Ratio",
]

SETUPS = [(1, 120), (1, 60), (5, 120)]


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def save_json(path, contents):
    path.write_text(
        json.dumps(contents, indent=2), encoding="utf-8"
    )


def main():
    config = load_config()
    settings = config["data"]

    require(
        config["split"] == {
            "train_end": "2019-12-31",
            "validation_end": "2022-12-31",
            "test_end": "2025-12-31",
        },
        "Unexpected final-evaluation split boundaries.",
    )
    require(
        config["forecast"]["target_column"] == "Close",
        "Expected Close as the target.",
    )

    train_end = pd.Timestamp(config["split"]["train_end"])
    validation_end = pd.Timestamp(config["split"]["validation_end"])

    ticker = settings["ticker"].strip().upper()
    stem = (
        f"{ticker}_{settings['start_date']}_"
        f"{settings['end_date_exclusive']}"
    )
    source = (
        PROJECT_ROOT / settings["processed_dir"]
        / f"{stem}_features.csv"
    )

    data = pd.read_csv(source, parse_dates=["Date"], index_col="Date")
    require(
        data.index.notna().all()
        and data.index.is_unique
        and data.index.is_monotonic_increasing,
        "Source dates must be valid, unique and chronological.",
    )
    require(
        all(column in data.columns for column in FEATURES),
        "Source is missing selected features.",
    )

    # Exclude final-test observations before constructing any arrays.
    data = data.loc[
        (data.index >= pd.Timestamp(settings["start_date"]))
        & (data.index <= validation_end),
        FEATURES,
    ].copy()

    values = data.to_numpy(dtype=float)
    require(len(data) > 125, "Insufficient development history.")
    require(
        np.isfinite(values[:, 0]).all() and (values[:, 0] > 0).all(),
        "Invalid Close values.",
    )
    require(not np.isinf(values).any(), "Infinite feature values.")

    complete = np.isfinite(values).all(axis=1)
    require(complete.any(), "No complete feature rows.")
    first_complete = int(np.flatnonzero(complete)[0])
    require(
        complete[first_complete:].all(),
        "Missing features occur after initial warm-up.",
    )

    # Fingerprint the development feature snapshot used here.
    development_hash = hashlib.sha256(
        data.to_csv(index_label="Date", float_format="%.17g").encode()
    ).hexdigest()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (
        PROJECT_ROOT / settings["processed_dir"]
        / "final_development" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    save_json(output_dir / "base_config_snapshot.json", config)
    summary = []

    for horizon, window in SETUPS:
        positions = []
        raw_windows = []

        for position in range(window - 1, len(data) - horizon):
            candidate = values[position - window + 1:position + 1]
            if np.isfinite(candidate).all():
                positions.append(position)
                raw_windows.append(candidate)

        require(bool(positions), "No complete windows.")
        positions = np.asarray(positions, dtype=int)
        X_raw = np.stack(raw_windows)

        metadata = pd.DataFrame({
            "Window_Start": data.index[positions - window + 1],
            "Forecast_Date": data.index[positions],
            "Target_Start_Date": data.index[positions + 1],
            "Target_Date": data.index[positions + horizon],
            "Actual_Close": values[positions + horizon, 0],
            "Naive_Prediction": values[positions, 0],
        })

        target_start = metadata["Target_Start_Date"]
        target_end = metadata["Target_Date"]

        masks = {
            "train": (target_end <= train_end).to_numpy(),
            "validation": (
                (target_start > train_end)
                & (target_end <= validation_end)
            ).to_numpy(),
        }

        require(
            all(mask.any() for mask in masks.values()),
            "An expected partition is empty.",
        )
        require(
            not (masks["train"] & masks["validation"]).any(),
            "Training and validation overlap.",
        )

        excluded = int(
            (~(masks["train"] | masks["validation"])).sum()
        )
        require(
            excluded == horizon - 1,
            "Unexpected number of boundary-crossing target spans.",
        )

        used_rows = np.zeros(len(data), dtype=bool)
        for position in positions[masks["train"]]:
            used_rows[position - window + 1:position + 1] = True

        require(
            data.index[used_rows].max() <= train_end,
            "Input preprocessing extends beyond training.",
        )

        scaler = StandardScaler()
        scaler.fit(values[used_rows, 1:])

        close_range = float(np.ptp(values[used_rows, 0]))
        target_range = float(np.ptp(
            metadata.loc[masks["train"], "Actual_Close"].to_numpy()
        ))
        require(
            close_range > 0 and target_range > 0,
            "Training price ranges must be positive.",
        )

        relative_close = (
            X_raw[:, :, :1] - X_raw[:, -1:, :1]
        ) / close_range

        extra_features = scaler.transform(
            X_raw[:, :, 1:].reshape(-1, len(FEATURES) - 1)
        ).reshape(len(metadata), window, len(FEATURES) - 1)

        X = np.concatenate(
            [relative_close, extra_features], axis=2
        ).astype(np.float32)

        changes = (
            metadata["Actual_Close"] - metadata["Naive_Prediction"]
        ).to_numpy()

        y = (changes / target_range).astype(np.float32).reshape(-1, 1)

        require(
            np.isfinite(X).all() and np.isfinite(y).all(),
            "Prepared arrays contain nonfinite values.",
        )
        require(
            np.allclose(X[:, -1, 0], 0),
            "Relative Close must end at zero.",
        )
        require(
            np.allclose(
                y.astype(float).ravel() * target_range
                + metadata["Naive_Prediction"].to_numpy(),
                metadata["Actual_Close"].to_numpy(),
                rtol=1e-6,
                atol=1e-5,
            ),
            "Target reconstruction failed.",
        )

        name = f"h{horizon}_w{window}"
        folder = output_dir / name
        folder.mkdir()

        setup_config = copy.deepcopy(config)
        setup_config["experiment"]["name"] = f"final_{name}"
        setup_config["forecast"].update({
            "feature_columns": FEATURES,
            "window_size": window,
            "horizon": horizon,
        })

        save_json(folder / "experiment.json", {
            "config": setup_config,
            "features": FEATURES,
            "window_size": window,
            "horizon": horizon,
            "target": "future_close_minus_current_close",
            "source": str(source.relative_to(PROJECT_ROOT)),
            "development_sha256": development_hash,
            "partitions": ["train", "validation"],
            "boundary_spans_excluded": excluded,
            "refit_on_validation": False,
        })

        save_json(folder / "preprocessing.json", {
            "feature_fit_first_date": str(
                data.index[used_rows].min().date()
            ),
            "feature_fit_last_date": str(
                data.index[used_rows].max().date()
            ),
            "feature_fit_rows": int(used_rows.sum()),
            "target_fit_first_date": str(
                target_end[masks["train"]].min().date()
            ),
            "target_fit_last_date": str(
                target_end[masks["train"]].max().date()
            ),
            "close_range": close_range,
            "target_range": target_range,
            "training_mean_change": float(changes[masks["train"]].mean()),
            "standard_scaler_columns": FEATURES[1:],
        })

        with (folder / "feature_scaler.pkl").open("wb") as file:
            pickle.dump(scaler, file)

        for split, mask in masks.items():
            subset = metadata.loc[mask].reset_index(drop=True)

            np.savez_compressed(
                folder / f"{split}.npz", X=X[mask], y=y[mask]
            )
            subset.to_csv(
                folder / f"{split}_metadata.csv", index=False
            )

            summary.append({
                "Dataset": name,
                "Split": split,
                "Samples": len(subset),
                "First_Target": str(subset["Target_Date"].min().date()),
                "Last_Target": str(subset["Target_Date"].max().date()),
            })

        logger.info(
            "%s: preprocessing and reconstruction PASS; "
            "excluded boundary spans=%s",
            name, excluded,
        )

    table = pd.DataFrame(summary)
    table.to_csv(output_dir / "summary.csv", index=False)

    print("\nFinal development datasets:")
    print(table.to_string(index=False))
    print(f"\nPrepared directory: {output_dir}")
    logger.info("Saved training and validation arrays only")


if __name__ == "__main__":
    run_logged(main, logger)