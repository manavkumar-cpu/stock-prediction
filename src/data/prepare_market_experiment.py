import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.download import PROJECT_ROOT, load_config
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("data.prepare_market_experiment")

FEATURES = [
    "Close",
    "Return_1",
    "RSI_14",
    "Volatility_20",
    "Relative_Volume_20",
    "Range_Ratio",
    "SP500_Return_1_Lag1",
    "NASDAQ_Return_1_Lag1",
    "VIX_Level_Lag1",
    "VIX_Change_1_Lag1",
]


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def main() -> None:
    config = load_config()
    settings = config["data"]
    window = config["forecast"]["window_size"]

    require(
        config["forecast"]["horizon"] == 1
        and config["forecast"]["target_column"] == "Close",
        "This experiment supports next-session Close only.",
    )
    require(window >= 2, "Window must contain at least two sessions.")

    ticker = settings["ticker"].strip().upper()
    stem = (
        f"{ticker}_{settings['start_date']}_"
        f"{settings['end_date_exclusive']}"
    )

    source = (
        PROJECT_ROOT
        / settings["processed_dir"]
        / f"{stem}_features_market.csv"
    )

    data = pd.read_csv(source, parse_dates=["Date"], index_col="Date")

    require(
        all(column in data.columns for column in FEATURES),
        "Source is missing selected feature columns.",
    )
    require(
        data.index.notna().all()
        and data.index.is_unique
        and data.index.is_monotonic_increasing,
        "Dates must be valid, unique, and chronological.",
    )

    data = data.loc[
        (data.index >= pd.Timestamp(settings["start_date"]))
        & (data.index < pd.Timestamp("2023-01-01")),
        FEATURES,
    ].copy()

    require(len(data) > window, "Insufficient development history.")

    values = data.to_numpy(dtype=float)
    close = values[:, 0]

    require(
        np.isfinite(close).all() and (close > 0).all(),
        "Close contains missing, infinite, or nonpositive values.",
    )
    require(
        not np.isinf(values).any(),
        "Features contain infinite values.",
    )

    # Only initial indicator warm-up NaNs are expected.
    complete = np.isfinite(values).all(axis=1)
    require(complete.any(), "No complete feature rows.")

    first_complete = int(np.flatnonzero(complete)[0])
    require(
        complete[first_complete:].all(),
        "Missing features occur after warm-up; investigate before proceeding.",
    )

    windows = []
    positions = []
    skipped = 0

    # Preserve session spacing: do not drop rows before windowing.
    for position in range(window - 1, len(data) - 1):
        candidate = values[position - window + 1:position + 1]

        if not np.isfinite(candidate).all():
            skipped += 1
            continue

        windows.append(candidate)
        positions.append(position)

    require(len(windows) > 0, "No complete windows available.")

    X_raw = np.stack(windows)
    positions = np.asarray(positions, dtype=int)

    metadata = pd.DataFrame({
        "Window_Start": data.index[positions - window + 1],
        "Forecast_Date": data.index[positions],
        "Target_Date": data.index[positions + 1],
        "Actual_Close": close[positions + 1],
        "Naive_Prediction": close[positions],
    })

    changes = (
        metadata["Actual_Close"] - metadata["Naive_Prediction"]
    ).to_numpy().reshape(-1, 1)

    years = metadata["Target_Date"].dt.year.to_numpy()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (
        PROJECT_ROOT
        / settings["processed_dir"]
        / "market_walk_forward"
        / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    experiment = {
        "config": config,
        "features": FEATURES,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "window_size": window,
        "target": "next_close_minus_current_close",
        "scoring_years": [2020, 2021, 2022],
        "warmup_windows_skipped": skipped,
        "refit_after_checkpoint_selection": False,
    }

    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )

    summary = []

    for score_year in experiment["scoring_years"]:
        masks = {
            "train": years <= score_year - 2,
            "selection": years == score_year - 1,
            "score": years == score_year,
        }

        require(
            all(mask.any() for mask in masks.values()),
            f"Empty partition in fold {score_year}.",
        )

        # Fit preprocessing on unique rows used in training windows.
        used_rows = np.zeros(len(data), dtype=bool)
        for position in positions[masks["train"]]:
            used_rows[position - window + 1:position + 1] = True

        feature_scaler = StandardScaler()
        feature_scaler.fit(values[used_rows, 1:])

        close_range = float(np.ptp(close[used_rows]))
        target_range = float(np.ptp(
            metadata.loc[masks["train"], "Actual_Close"].to_numpy()
        ))

        require(
            close_range > 0 and target_range > 0,
            "Training price ranges must be positive.",
        )

        # Center Close on the last observed Close of each window.
        relative_close = (
            X_raw[:, :, :1] - X_raw[:, -1:, :1]
        ) / close_range

        extra_features = feature_scaler.transform(
            X_raw[:, :, 1:].reshape(-1, len(FEATURES) - 1)
        ).reshape(len(X_raw), window, len(FEATURES) - 1)

        X = np.concatenate(
            [relative_close, extra_features], axis=2
        ).astype(np.float32)

        y = (changes / target_range).astype(np.float32)

        require(
            np.isfinite(X).all() and np.isfinite(y).all(),
            "Prepared arrays contain nonfinite values.",
        )
        require(
            np.allclose(X[:, -1, 0], 0),
            "Centered Close must end at zero.",
        )
        require(
            np.allclose(
                y.astype(np.float64).ravel() * target_range
                + metadata["Naive_Prediction"].to_numpy(),
                metadata["Actual_Close"].to_numpy(),
                rtol=1e-6,
                atol=1e-5,
            ),
            "Target reconstruction failed.",
        )

        fold_dir = output_dir / str(score_year)
        fold_dir.mkdir()

        with (fold_dir / "feature_scaler.pkl").open("wb") as file:
            pickle.dump(feature_scaler, file)

        fold_info = {
            "score_year": score_year,
            "selection_year": score_year - 1,
            "fit_target_end": str(
                metadata.loc[masks["train"], "Target_Date"].max().date()
            ),
            "feature_fit_first_date": str(data.index[used_rows][0].date()),
            "feature_fit_last_date": str(data.index[used_rows][-1].date()),
            "feature_fit_rows": int(used_rows.sum()),
            "close_range": close_range,
            "target_range": target_range,
        }

        (fold_dir / "preprocessing.json").write_text(
            json.dumps(fold_info, indent=2), encoding="utf-8"
        )

        for split, mask in masks.items():
            subset = metadata.loc[mask].reset_index(drop=True)

            np.savez_compressed(
                fold_dir / f"{split}.npz",
                X=X[mask],
                y=y[mask],
            )
            subset.to_csv(
                fold_dir / f"{split}_metadata.csv", index=False
            )

            summary.append({
                "Score_Year": score_year,
                "Split": split,
                "Samples": len(subset),
                "First_Target": str(subset["Target_Date"].min().date()),
                "Last_Target": str(subset["Target_Date"].max().date()),
            })

        logger.info("Fold %s: preprocessing checks PASS", score_year)

    summary_table = pd.DataFrame(summary)
    summary_table.to_csv(output_dir / "summary.csv", index=False)

    print("\nSelected features:", FEATURES)
    print("Windows skipped for feature warm-up:", skipped)
    print("\nFold summary:")
    print(summary_table.to_string(index=False))
    print(f"\nPrepared data directory: {output_dir}")
    logger.info(
        "Prepared ten-feature arrays with matching six-feature controls"
    )


if __name__ == "__main__":
    run_logged(main, logger)