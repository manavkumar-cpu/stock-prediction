import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import PROJECT_ROOT
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.regimes")


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(actual, predicted):
    error = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_percent": float(
            100 * np.mean(np.abs(error / actual))
        ),
        "Mean_Error": float(np.mean(error)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    experiment = read_json(run_dir / "experiment.json")
    config = experiment["config"]
    settings = config["data"]
    horizon = int(config["forecast"]["horizon"])

    require(
        horizon in [1, 5]
        and experiment["scoring_years"] == [2020, 2021, 2022],
        "Expected development forecasts with horizon one or five.",
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
        "Invalid development closing prices.",
    )

    # All rolling calculations look backward and include today's Close.
    conditions = pd.DataFrame(index=close.index)
    conditions["Close"] = close
    conditions["SMA_50"] = close.rolling(
        50, min_periods=50
    ).mean()

    returns = close.pct_change(fill_method=None)
    conditions["Volatility_20"] = returns.rolling(
        20, min_periods=20
    ).std(ddof=1)

    predictions = pd.read_csv(
        run_dir / "all_predictions.csv",
        parse_dates=["Forecast_Date", "Target_Date"],
    )

    require(
        predictions["Target_Date"].notna().all()
        and predictions["Target_Date"].is_unique
        and predictions["Target_Date"].is_monotonic_increasing
        and predictions["Target_Date"].dt.year.isin(
            [2020, 2021, 2022]
        ).all(),
        "Invalid development prediction dates.",
    )

    expected_samples = 756 if horizon == 1 else 744
    require(
        len(predictions) == expected_samples,
        "Unexpected prediction count.",
    )

    model_columns = [
        column
        for column in ["Six_feature_LSTM", "Attention_LSTM"]
        if column in predictions.columns
    ]
    require(model_columns, "No supported LSTM predictions found.")

    forecast_positions = close.index.get_indexer(
        pd.DatetimeIndex(predictions["Forecast_Date"])
    )
    target_positions = close.index.get_indexer(
        pd.DatetimeIndex(predictions["Target_Date"])
    )

    require(
        (forecast_positions >= 0).all()
        and (target_positions >= 0).all()
        and np.all(
            target_positions - forecast_positions == horizon
        ),
        "Forecast dates do not match the stated horizon.",
    )

    np.testing.assert_allclose(
        predictions["Naive_Prediction"],
        close.iloc[forecast_positions],
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        predictions["Actual_Close"],
        close.iloc[target_positions],
        rtol=1e-8,
        atol=1e-8,
    )

    assigned = conditions.reindex(
        pd.DatetimeIndex(predictions["Forecast_Date"])
    ).reset_index(drop=True)

    require(
        np.isfinite(assigned.to_numpy()).all(),
        "Missing forecast-date regime inputs.",
    )

    predictions["Naive"] = predictions["Naive_Prediction"]
    predictions["SMA_50"] = assigned["SMA_50"].to_numpy()
    predictions["Volatility_20"] = (
        assigned["Volatility_20"].to_numpy()
    )
    predictions["Trend"] = np.where(
        assigned["Close"].to_numpy()
        >= assigned["SMA_50"].to_numpy(),
        "Above_or_at_SMA50",
        "Below_SMA50",
    )
    predictions["Volatility_Regime"] = ""
    predictions["Volatility_Threshold"] = np.nan

    thresholds = []

    for year in [2020, 2021, 2022]:
        preprocessing = read_json(
            run_dir / str(year) / "preprocessing.json"
        )
        fit_start = pd.Timestamp(
            preprocessing["feature_fit_first_date"]
        )
        fit_end = pd.Timestamp(
            preprocessing["feature_fit_last_date"]
        )

        require(
            fit_start <= fit_end and fit_end.year <= year - 2,
            "Regime threshold fitting extends beyond training.",
        )

        training_volatility = conditions.loc[
            fit_start:fit_end, "Volatility_20"
        ].dropna()

        require(
            len(training_volatility) > 100,
            "Insufficient training volatility observations.",
        )

        threshold = float(training_volatility.quantile(0.75))
        mask = predictions["Target_Date"].dt.year == year

        predictions.loc[mask, "Volatility_Regime"] = np.where(
            predictions.loc[mask, "Volatility_20"] > threshold,
            "High_volatility",
            "Lower_volatility",
        )
        predictions.loc[mask, "Volatility_Threshold"] = threshold

        thresholds.append({
            "Score_Year": year,
            "Fit_Start": str(fit_start.date()),
            "Fit_End": str(fit_end.date()),
            "Fit_Rows": len(training_volatility),
            "Volatility_75th_Percentile": threshold,
        })

    predictions["Combined_Regime"] = (
        predictions["Trend"]
        + " / "
        + predictions["Volatility_Regime"]
    )

    models = ["Naive", *model_columns]
    require(
        np.isfinite(
            predictions[["Actual_Close", *models]].to_numpy()
        ).all(),
        "Nonfinite prices or predictions.",
    )

    rows = []

    # Include yearly results so pooled groups cannot hide year effects.
    for period in ["Pooled", 2020, 2021, 2022]:
        period_data = (
            predictions
            if period == "Pooled"
            else predictions.loc[
                predictions["Target_Date"].dt.year == period
            ]
        )

        for grouping in [
            "Trend", "Volatility_Regime", "Combined_Regime"
        ]:
            for regime, group in period_data.groupby(
                grouping, sort=True
            ):
                actual = group["Actual_Close"].to_numpy()
                naive = metrics(
                    actual, group["Naive"].to_numpy()
                )

                for model in models:
                    result = metrics(
                        actual, group[model].to_numpy()
                    )
                    rows.append({
                        "Period": period,
                        "Horizon": horizon,
                        "Grouping": grouping,
                        "Regime": regime,
                        "Model": model,
                        "Samples": len(group),
                        **result,
                        "MAE_Improvement_Percent": (
                            100 * (1 - result["MAE"] / naive["MAE"])
                            if naive["MAE"] > 0 else np.nan
                        ),
                        "RMSE_Improvement_Percent": (
                            100 * (1 - result["RMSE"] / naive["RMSE"])
                            if naive["RMSE"] > 0 else np.nan
                        ),
                    })

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    output_dir = run_dir / "regime_analysis" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "regime_metrics.csv", index=False)
    predictions.to_csv(
        output_dir / "predictions_with_regimes.csv", index=False
    )
    pd.DataFrame(thresholds).to_csv(
        output_dir / "thresholds.csv", index=False
    )

    (output_dir / "definitions.json").write_text(
        json.dumps({
            "source_run": str(run_dir.resolve()),
            "horizon": horizon,
            "trend": "Forecast-date Close >= trailing SMA50",
            "volatility": "Trailing 20-session return standard deviation",
            "high_volatility": "Above fold-training 75th percentile",
            "threshold_fit_period": "Saved feature-scaler fitting dates",
            "purpose": "Descriptive development error analysis",
        }, indent=2),
        encoding="utf-8",
    )

    print("\nTraining-derived volatility thresholds:")
    print(pd.DataFrame(thresholds).to_string(index=False))

    print("\nPooled regime results:")
    print(
        report.loc[
            (report["Period"] == "Pooled")
            & (report["Grouping"] != "Combined_Regime"),
            [
                "Grouping", "Regime", "Model", "Samples",
                "MAE", "RMSE", "Mean_Error",
                "RMSE_Improvement_Percent",
            ],
        ].to_string(index=False, float_format="%.6f")
    )

    logger.info("Regime alignment checks PASS")
    logger.info("Saved regime analysis: %s", output_dir)


if __name__ == "__main__":
    run_logged(main, logger)