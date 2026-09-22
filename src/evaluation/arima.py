import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.arima.model import ARIMA

from src.data.download import PROJECT_ROOT
from src.evaluation.metrics import regression_metrics
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.arima")

ORDERS = [
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
]


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(folder, split, close, year):
    table = pd.read_csv(
        folder / f"{split}_metadata.csv",
        parse_dates=["Window_Start", "Forecast_Date", "Target_Date"],
    )

    dates = pd.DatetimeIndex(table["Target_Date"])
    forecasts = pd.DatetimeIndex(table["Forecast_Date"])

    require(
        len(table) > 1
        and dates.notna().all()
        and dates.is_unique
        and dates.is_monotonic_increasing,
        f"{split}: invalid target dates.",
    )

    positions = close.index.get_indexer(dates)
    forecast_positions = close.index.get_indexer(forecasts)

    require(
        (positions > 0).all()
        and (forecast_positions >= 0).all()
        and np.array_equal(positions, forecast_positions + 1)
        and (np.diff(positions) == 1).all(),
        f"{split}: dates are not consecutive next-session examples.",
    )

    if split == "train":
        correct_year = dates.year <= year - 2
    elif split == "selection":
        correct_year = dates.year == year - 1
    else:
        correct_year = dates.year == year

    require(correct_year.all(), f"{split}: unexpected target period.")

    np.testing.assert_allclose(
        table["Actual_Close"],
        close.iloc[positions],
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        table["Naive_Prediction"],
        close.iloc[forecast_positions],
        rtol=1e-8,
        atol=1e-8,
    )

    return table


def forecast_sequentially(result, table, last_observed_date):
    predictions = []

    for row in table.itertuples(index=False):
        require(
            row.Forecast_Date == last_observed_date,
            "Model state is not aligned with the forecast date.",
        )

        # Forecast before revealing this session's actual Close.
        prediction = float(
            np.asarray(result.forecast(steps=1)).ravel()[0]
        )
        require(np.isfinite(prediction), "Nonfinite ARIMA forecast.")
        predictions.append(prediction)

        # Update state only; fitted coefficients remain unchanged.
        result = result.extend(
            np.asarray([row.Actual_Close], dtype=float)
        )
        last_observed_date = row.Target_Date

    return np.asarray(predictions), result, last_observed_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    experiment = read_json(data_dir / "experiment.json")
    config = experiment["config"]
    settings = config["data"]

    require(
        experiment["scoring_years"] == [2020, 2021, 2022]
        and config["forecast"]["horizon"] == 1,
        "Expected next-session development folds for 2020–2022.",
    )

    ticker = settings["ticker"].strip().upper()
    stem = (
        f"{ticker}_{settings['start_date']}_"
        f"{settings['end_date_exclusive']}"
    )
    raw_path = (
        PROJECT_ROOT / settings["raw_dir"] / f"{stem}_raw.csv"
    )

    raw = pd.read_csv(
        raw_path, parse_dates=["Date"], index_col="Date"
    )
    close = raw.loc[
        (raw.index >= pd.Timestamp(settings["start_date"]))
        & (raw.index < pd.Timestamp("2023-01-01")),
        "Close",
    ].astype(float)

    require(
        close.index.notna().all()
        and close.index.is_unique
        and close.index.is_monotonic_increasing
        and np.isfinite(close).all()
        and (close > 0).all(),
        "Invalid development price series.",
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = (
        PROJECT_ROOT / config["paths"]["artifacts"]
        / "arima_walk_forward" / stamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "prepared_data": str(data_dir.resolve()),
        "source": str(raw_path.relative_to(PROJECT_ROOT)),
        "orders": ORDERS,
        "trend": "n",
        "selection_metric": "RMSE",
        "refit_after_selection": False,
        "state_update": "after_each_observed_close",
        "statsmodels_version": statsmodels.__version__,
        "experiment": experiment,
    }
    (run_dir / "experiment.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    fold_metrics = []
    prediction_tables = []

    for year in experiment["scoring_years"]:
        source_fold = data_dir / str(year)
        fold_dir = run_dir / str(year)
        fold_dir.mkdir()

        train = load_metadata(source_fold, "train", close, year)
        selection = load_metadata(
            source_fold, "selection", close, year
        )

        fit_start = train["Window_Start"].min()
        fit_end = train["Target_Date"].max()
        history = close.loc[fit_start:fit_end].to_numpy()

        require(
            selection["Forecast_Date"].iloc[0] == fit_end,
            "Training and selection periods do not connect.",
        )

        candidate_rows = []
        best = None

        for order in ORDERS:
            name = f"ARIMA{order}"
            logger.info("%s: fitting %s", year, name)

            try:
                fitted = ARIMA(
                    history,
                    order=order,
                    trend="n",
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(method_kwargs={"maxiter": 500})

                require(
                    bool(fitted.mle_retvals.get("converged", False)),
                    f"{name}: optimizer did not converge.",
                )

                predicted, updated, last_date = forecast_sequentially(
                    fitted, selection, fit_end
                )
                metrics = regression_metrics(
                    selection["Actual_Close"].to_numpy(),
                    predicted,
                )

            except (
                ValueError,
                np.linalg.LinAlgError,
                DataValidationError,
            ) as error:
                candidate_rows.append({
                    "Order": str(order),
                    "Status": "FAILED",
                    "Reason": str(error),
                })
                pd.DataFrame(candidate_rows).to_csv(
                    fold_dir / "selection_candidates.csv", index=False
                )
                logger.warning("%s: %s failed: %s", year, name, error)
                continue

            if order == (0, 1, 0):
                np.testing.assert_allclose(
                    predicted,
                    selection["Naive_Prediction"],
                    rtol=1e-7,
                    atol=1e-6,
                )
                logger.info("%s: random-walk/naive check PASS", year)

            candidate_rows.append({
                "Order": str(order),
                "Status": "OK",
                **metrics,
            })

            candidate_predictions = selection.copy()
            candidate_predictions["Prediction"] = predicted
            candidate_predictions.to_csv(
                fold_dir / f"selection_{order[0]}_1_{order[2]}.csv",
                index=False,
            )

            if best is None or metrics["RMSE"] < best["rmse"]:
                best = {
                    "order": order,
                    "rmse": metrics["RMSE"],
                    "result": updated,
                    "last_date": last_date,
                }

            pd.DataFrame(candidate_rows).to_csv(
                fold_dir / "selection_candidates.csv", index=False
            )

        require(best is not None, f"{year}: every candidate failed.")

        choice = {
            "score_year": year,
            "order": best["order"],
            "selection_rmse": best["rmse"],
            "fit_start": str(fit_start.date()),
            "fit_end": str(fit_end.date()),
            "fit_observations": len(history),
        }
        (fold_dir / "selected_model.json").write_text(
            json.dumps(choice, indent=2), encoding="utf-8"
        )
        best["result"].save(
            str(fold_dir / "model_before_scoring.pkl")
        )

        # Scoring starts only after order selection is complete.
        score = load_metadata(source_fold, "score", close, year)
        predicted, _, _ = forecast_sequentially(
            best["result"], score, best["last_date"]
        )

        score["ARIMA"] = predicted
        score["Naive"] = score["Naive_Prediction"]
        score.to_csv(fold_dir / "predictions.csv", index=False)
        prediction_tables.append(score)

        for name in ["Naive", "ARIMA"]:
            actual = score["Actual_Close"].to_numpy()
            estimates = score[name].to_numpy()
            fold_metrics.append({
                "Score_Year": year,
                "Model": name,
                "Samples": len(score),
                **regression_metrics(actual, estimates),
                "Mean_Error": float(np.mean(estimates - actual)),
            })

        pd.DataFrame(fold_metrics).to_csv(
            run_dir / "fold_metrics.csv", index=False
        )
        logger.info(
            "%s: selected %s using selection-year RMSE %.6f",
            year, best["order"], best["rmse"],
        )

    combined = pd.concat(prediction_tables, ignore_index=True)
    require(
        len(combined) == 756
        and combined["Target_Date"].is_unique,
        "Unexpected pooled scoring dates.",
    )
    combined.to_csv(run_dir / "all_predictions.csv", index=False)

    pooled_rows = []
    for name in ["Naive", "ARIMA"]:
        actual = combined["Actual_Close"].to_numpy()
        estimates = combined[name].to_numpy()
        pooled_rows.append({
            "Model": name,
            "Samples": len(combined),
            **regression_metrics(actual, estimates),
            "Mean_Error": float(np.mean(estimates - actual)),
        })

    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(run_dir / "pooled_metrics.csv", index=False)

    print("\nResults by scoring year:")
    print(pd.DataFrame(fold_metrics).to_string(index=False))
    print("\nPooled development results:")
    print(pooled.to_string(index=False))
    logger.info("Saved ARIMA comparison: %s", run_dir)


if __name__ == "__main__":
    run_logged(main, logger)