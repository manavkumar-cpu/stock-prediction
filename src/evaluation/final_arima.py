import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from src.data.download import PROJECT_ROOT
from src.evaluation.metrics import regression_metrics
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.final_arima")

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


def save_json(path, contents):
    path.write_text(
        json.dumps(contents, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_dir = data_dir.resolve()

    experiment = read_json(data_dir / "experiment.json")
    config = experiment["config"]
    settings = config["data"]

    require(
        experiment["horizon"] == 1
        and experiment["window_size"] == 120,
        "Expected final h1_w120 reference data.",
    )

    date_columns = [
        "Window_Start", "Forecast_Date",
        "Target_Start_Date", "Target_Date",
    ]
    train_meta = pd.read_csv(
        data_dir / "train_metadata.csv",
        parse_dates=date_columns,
    )
    val_meta = pd.read_csv(
        data_dir / "validation_metadata.csv",
        parse_dates=date_columns,
    )

    for name, table in (("train", train_meta), ("validation", val_meta)):
        require(
            len(table) > 1
            and table[date_columns].notna().all().all()
            and table["Target_Date"].is_unique
            and table["Target_Date"].is_monotonic_increasing,
            f"{name}: invalid metadata dates.",
        )

    require(
        train_meta["Target_Date"].max()
        == pd.Timestamp("2019-12-31")
        and (val_meta["Target_Date"] > "2019-12-31").all()
        and (val_meta["Target_Date"] <= "2022-12-31").all()
        and len(val_meta) == 756,
        "Unexpected training or validation period.",
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
        "Invalid raw date index.",
    )

    # Preserve the development ARIMA convention: begin at the
    # earliest input date in the reference training windows.
    fit_start = train_meta["Window_Start"].min()
    fit_end = train_meta["Target_Date"].max()

    close = raw.loc[
        (raw.index >= fit_start)
        & (raw.index <= pd.Timestamp("2022-12-31")),
        "Close",
    ]
    require(
        np.isfinite(close.to_numpy()).all() and (close > 0).all(),
        "Invalid development Close values.",
    )

    training = close.loc[close.index <= fit_end]
    validation = close.loc[close.index > fit_end]

    require(
        np.array_equal(
            validation.index.to_numpy(),
            val_meta["Target_Date"].to_numpy(),
        ),
        "Raw validation dates differ from reference targets.",
    )

    positions = close.index.get_indexer(val_meta["Target_Date"])
    require(
        (positions > 0).all()
        and np.array_equal(
            close.index[positions - 1].to_numpy(),
            val_meta["Forecast_Date"].to_numpy(),
        ),
        "Forecast dates are not the preceding observed sessions.",
    )

    actual = validation.to_numpy(dtype=float)
    naive = val_meta["Naive_Prediction"].to_numpy(dtype=float)

    require(
        np.allclose(actual, val_meta["Actual_Close"].to_numpy())
        and np.allclose(naive, close.to_numpy()[positions - 1]),
        "Raw/reference price alignment failed.",
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = (
        PROJECT_ROOT / config["paths"]["artifacts"]
        / "final_arima" / stamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(
        data_dir / "experiment.json", run_dir / "experiment.json"
    )
    save_json(run_dir / "run_info.json", {
        "reference_data": str(data_dir),
        "source": str(source.relative_to(PROJECT_ROOT)),
        "statsmodels_version": statsmodels.__version__,
        "candidate_orders": ORDERS,
        "trend": "n",
        "fit_start": str(training.index.min().date()),
        "fit_end": str(training.index.max().date()),
        "fit_rows": len(training),
        "validation_start": str(validation.index.min().date()),
        "validation_end": str(validation.index.max().date()),
        "selection": "minimum sequential validation RMSE",
        "refit_on_validation": False,
        "state_updates": "after observing each validation target",
        "test_evaluated": False,
    })

    rows = []
    successful = {}
    candidate_predictions = val_meta.copy()

    for order in ORDERS:
        name = "ARIMA_" + "_".join(map(str, order))
        logger.info("Fitting %s", name)

        try:
            fitted = ARIMA(
                training.to_numpy(dtype=float),
                order=order,
                trend="n",
            ).fit(method_kwargs={"maxiter": 500})
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append({
                "Candidate": name,
                "Status": "FAILED",
                "Reason": str(error),
            })
            pd.DataFrame(rows).to_csv(
                run_dir / "candidate_metrics.csv", index=False
            )
            logger.warning("%s fit failed: %s", name, error)
            if order == (0, 1, 0):
                raise
            continue

        converged = bool(fitted.mle_retvals.get("converged", False))
        if not converged or not np.isfinite(fitted.params).all():
            rows.append({
                "Candidate": name,
                "Status": "FAILED",
                "Reason": "Nonconvergence or nonfinite parameters",
            })
            pd.DataFrame(rows).to_csv(
                run_dir / "candidate_metrics.csv", index=False
            )
            require(
                order != (0, 1, 0),
                "Random-walk reference model failed to fit.",
            )
            logger.warning("%s excluded: invalid fit", name)
            continue

        fitted.save(str(run_dir / f"{name}_training.pkl"))
        initial_params = np.asarray(fitted.params).copy()
        state = fitted
        predicted = []

        for observed in actual:
            forecast = float(np.asarray(state.forecast(steps=1))[0])
            require(np.isfinite(forecast), f"{name}: nonfinite forecast.")
            predicted.append(forecast)

            # The actual is incorporated only after its forecast.
            state = state.extend(np.asarray([observed], dtype=float))

        predicted = np.asarray(predicted)
        require(
            np.array_equal(initial_params, np.asarray(state.params)),
            f"{name}: coefficients changed during validation.",
        )

        if order == (0, 1, 0):
            require(
                np.allclose(predicted, naive, rtol=1e-7, atol=1e-6),
                "ARIMA(0,1,0) does not reproduce naive.",
            )
            logger.info("Random-walk/naive check PASS")

        state_path = run_dir / f"{name}_validation_state.pkl"
        state.save(str(state_path))

        reloaded = ARIMAResults.load(str(state_path))
        require(
            np.allclose(
                state.forecast(steps=1),
                reloaded.forecast(steps=1),
                rtol=1e-10,
                atol=1e-10,
            )
            and np.array_equal(initial_params, np.asarray(reloaded.params)),
            f"{name}: saved-state reload check failed.",
        )

        metrics = regression_metrics(actual, predicted)
        rows.append({
            "Candidate": name,
            "Status": "PASS",
            **metrics,
        })
        successful[name] = {
            "order": order,
            "rmse": metrics["RMSE"],
            "state_path": state_path,
            "predictions": predicted,
        }
        candidate_predictions[name] = predicted

        pd.DataFrame(rows).to_csv(
            run_dir / "candidate_metrics.csv", index=False
        )
        logger.info(
            "%s: validation RMSE=%.6f; fixed coefficients and reload PASS",
            name, metrics["RMSE"],
        )

    require(bool(successful), "No valid ARIMA candidates.")

    # Candidate order provides deterministic tie-breaking.
    selected_name = min(successful, key=lambda n: successful[n]["rmse"])
    selected = successful[selected_name]

    selected_path = run_dir / "selected_validation_state.pkl"
    shutil.copy2(selected["state_path"], selected_path)

    save_json(run_dir / "selected_model.json", {
        "model": "ARIMA_h1",
        "order": selected["order"],
        "trend": "n",
        "checkpoint": selected_path.name,
        "checkpoint_sha256": hashlib.sha256(
            selected_path.read_bytes()
        ).hexdigest(),
        "validation_rmse": selected["rmse"],
        "coefficient_fit_end": str(fit_end.date()),
        "state_as_of": str(validation.index.max().date()),
        "horizon": 1,
        "model_output": "price_usd",
    })

    candidate_predictions.to_csv(
        run_dir / "candidate_predictions.csv", index=False
    )

    predictions = {
        "Naive": naive,
        "ARIMA_h1": selected["predictions"],
    }
    metric_rows = []
    for name, predicted in predictions.items():
        val_meta[name] = predicted
        metric_rows.append({
            "Model": name,
            "Samples": len(actual),
            **regression_metrics(actual, predicted),
            "Mean_Error": float(np.mean(predicted - actual)),
        })

    val_meta.to_csv(run_dir / "validation_predictions.csv", index=False)
    table = pd.DataFrame(metric_rows)
    table.to_csv(run_dir / "validation_metrics.csv", index=False)

    print("\nCandidate results:")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.6f"))
    print(f"\nSelected order: {selected['order']}")
    print("\nValidation comparison:")
    print(table.to_string(index=False, float_format="%.6f"))

    logger.info("Saved final ARIMA selection: %s", run_dir)
    logger.info("Final-test data has not been evaluated")


if __name__ == "__main__":
    run_logged(main, logger)