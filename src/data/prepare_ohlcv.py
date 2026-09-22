import argparse
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.download import PROJECT_ROOT
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("data.prepare_ohlcv")

# Keep Close first for compatibility with our training checks.
FEATURES = ["Close", "Open", "High", "Low", "Volume"]
DATE_COLUMNS = ["Window_Start", "Forecast_Date", "Target_Date"]


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir).expanduser()
    if not reference_dir.is_absolute():
        reference_dir = PROJECT_ROOT / reference_dir

    experiment = read_json(reference_dir / "experiment.json")
    config = experiment["config"]
    settings = config["data"]
    window = experiment["window_size"]

    require(
        window == 120
        and config["forecast"]["horizon"] == 1
        and experiment["target"] == "next_close_minus_current_close",
        "Expected 120-session windows and next-session price changes.",
    )
    require(
        experiment["scoring_years"] == [2020, 2021, 2022],
        "Expected development scoring years 2020–2022.",
    )
    require(
        experiment["features"] == [
            "Close", "Return_1", "RSI_14", "Volatility_20",
            "Relative_Volume_20", "Range_Ratio",
        ],
        "Unexpected reference feature order.",
    )

    ticker = settings["ticker"].strip().upper()
    stem = (
        f"{ticker}_{settings['start_date']}_"
        f"{settings['end_date_exclusive']}"
    )
    source = PROJECT_ROOT / settings["raw_dir"] / f"{stem}_raw.csv"

    raw = pd.read_csv(source, parse_dates=["Date"], index_col="Date")

    require(
        raw.index.notna().all()
        and raw.index.is_unique
        and raw.index.is_monotonic_increasing,
        "Raw dates must be valid, unique and chronological.",
    )
    require(
        all(column in raw.columns for column in FEATURES),
        "Raw data is missing OHLCV columns.",
    )

    data = raw.loc[
        (raw.index >= pd.Timestamp(settings["start_date"]))
        & (raw.index < pd.Timestamp("2023-01-01")),
        FEATURES,
    ].copy()

    values = data.to_numpy(dtype=float)
    require(np.isfinite(values).all(), "Nonfinite development OHLCV.")
    require((values[:, :4] > 0).all(), "Nonpositive prices.")
    require((values[:, 4] >= 0).all(), "Negative volume.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (
        PROJECT_ROOT / settings["processed_dir"]
        / "ohlcv_walk_forward" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    output_experiment = dict(experiment)
    output_experiment.update({
        "features": FEATURES,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "reference_data": str(reference_dir.resolve()),
        "price_transform": (
            "OHLC minus last observed Close, divided by reference "
            "training Close range"
        ),
        "volume_transform": "StandardScaler on unique training input rows",
        "sample_policy": "Reuse all reference partitions and target dates",
    })

    (output_dir / "experiment.json").write_text(
        json.dumps(output_experiment, indent=2), encoding="utf-8"
    )
    # Keep the exact reference manifest for provenance.
    shutil.copy2(
        reference_dir / "experiment.json",
        output_dir / "reference_experiment.json",
    )

    summary = []

    for year in experiment["scoring_years"]:
        source_fold = reference_dir / str(year)
        fold_dir = output_dir / str(year)
        fold_dir.mkdir()

        preprocessing = read_json(source_fold / "preprocessing.json")
        close_range = float(preprocessing["close_range"])
        target_range = float(preprocessing["target_range"])

        require(
            np.isfinite([close_range, target_range]).all()
            and close_range > 0 and target_range > 0,
            "Invalid reference price ranges.",
        )

        partitions = {}
        used_rows = np.zeros(len(data), dtype=bool)

        for split in ("train", "selection", "score"):
            metadata = pd.read_csv(
                source_fold / f"{split}_metadata.csv",
                parse_dates=DATE_COLUMNS,
            )
            dates = metadata["Target_Date"]

            require(
                len(metadata) > 0
                and metadata[DATE_COLUMNS].notna().all().all()
                and dates.is_unique
                and dates.is_monotonic_increasing,
                f"{year}/{split}: invalid metadata dates.",
            )

            valid_years = (
                dates.dt.year <= year - 2 if split == "train"
                else dates.dt.year == (
                    year - 1 if split == "selection" else year
                )
            )
            require(valid_years.all(), f"{year}/{split}: wrong partition.")

            positions = data.index.get_indexer(metadata["Forecast_Date"])
            targets = data.index.get_indexer(dates)

            require(
                (positions >= window - 1).all()
                and (targets >= 0).all()
                and (targets == positions + 1).all(),
                f"{year}/{split}: next-session alignment failed.",
            )

            rows = (
                positions[:, None]
                + np.arange(-window + 1, 1)[None, :]
            )
            require(
                np.array_equal(
                    data.index[rows[:, 0]].to_numpy(),
                    metadata["Window_Start"].to_numpy(),
                ),
                f"{year}/{split}: window start mismatch.",
            )

            require(
                np.allclose(
                    values[positions, 0],
                    metadata["Naive_Prediction"].to_numpy(),
                )
                and np.allclose(
                    values[targets, 0],
                    metadata["Actual_Close"].to_numpy(),
                ),
                f"{year}/{split}: raw/reference price mismatch.",
            )

            partitions[split] = (metadata, rows)
            if split == "train":
                used_rows[rows.ravel()] = True

        fit_dates = data.index[used_rows]
        require(
            fit_dates[-1].year <= year - 2,
            "Volume preprocessing extends beyond training.",
        )
        require(
            str(fit_dates[0].date())
            == preprocessing["feature_fit_first_date"]
            and str(fit_dates[-1].date())
            == preprocessing["feature_fit_last_date"]
            and int(used_rows.sum()) == preprocessing["feature_fit_rows"],
            "Training input coverage differs from reference.",
        )

        volume_scaler = StandardScaler()
        volume_scaler.fit(values[used_rows, 4:5])

        # This scaler applies only to Volume; OHLC uses close_range.
        with (fold_dir / "feature_scaler.pkl").open("wb") as file:
            pickle.dump(volume_scaler, file)

        preprocessing.update({
            "features": FEATURES,
            "standard_scaler_columns": ["Volume"],
            "relative_price_columns": FEATURES[:4],
        })
        (fold_dir / "preprocessing.json").write_text(
            json.dumps(preprocessing, indent=2), encoding="utf-8"
        )

        for split, (metadata, rows) in partitions.items():
            raw_windows = values[rows]
            price_channels = (
                raw_windows[:, :, :4] - raw_windows[:, -1:, :1]
            ) / close_range

            volume_channel = volume_scaler.transform(
                raw_windows[:, :, 4:5].reshape(-1, 1)
            ).reshape(len(metadata), window, 1)

            X = np.concatenate(
                [price_channels, volume_channel], axis=2
            ).astype(np.float32)

            with np.load(
                source_fold / f"{split}.npz", allow_pickle=False
            ) as reference:
                reference_X = reference["X"]
                y = reference["y"].copy()

            require(
                X.shape == (len(metadata), window, 5)
                and y.shape == (len(metadata), 1)
                and np.isfinite(X).all()
                and np.isfinite(y).all(),
                f"{year}/{split}: invalid prepared arrays.",
            )
            require(
                np.array_equal(X[:, :, 0], reference_X[:, :, 0]),
                f"{year}/{split}: Close channel differs from reference.",
            )
            require(
                np.allclose(
                    y.astype(float).ravel() * target_range
                    + metadata["Naive_Prediction"].to_numpy(),
                    metadata["Actual_Close"].to_numpy(),
                    rtol=1e-6,
                    atol=1e-5,
                ),
                f"{year}/{split}: target reconstruction failed.",
            )

            np.savez_compressed(fold_dir / f"{split}.npz", X=X, y=y)
            metadata.to_csv(
                fold_dir / f"{split}_metadata.csv", index=False
            )

            summary.append({
                "Score_Year": year,
                "Split": split,
                "Samples": len(metadata),
                "First_Target": str(metadata["Target_Date"].min().date()),
                "Last_Target": str(metadata["Target_Date"].max().date()),
            })

        logger.info(
            "Fold %s: matching dates, targets and Close channel PASS",
            year,
        )

    table = pd.DataFrame(summary)
    table.to_csv(output_dir / "summary.csv", index=False)

    print("\nFeatures:", FEATURES)
    print("\nFold summary:")
    print(table.to_string(index=False))
    print(f"\nPrepared OHLCV directory: {output_dir}")
    logger.info("Training-only volume scaling checks PASS")


if __name__ == "__main__":
    run_logged(main, logger)