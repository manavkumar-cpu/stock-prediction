from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.download import PROJECT_ROOT
from src.evaluation.metrics import regression_metrics
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.compare_models")

RUNS = {
    "LSTM_w120": (
        "artifacts/attention_walk_forward/w120/"
        "20260919T185751_441646Z",
        "Six_feature_LSTM",
        "120 sessions; common training dates",
    ),
    "Attention_LSTM_w120": (
        "artifacts/attention_walk_forward/w120/"
        "20260919T185751_441646Z",
        "Attention_LSTM",
        "120 sessions; common training dates",
    ),
    "XGBoost_w120": (
        "artifacts/xgboost_walk_forward/"
        "20260919T183633_049138Z",
        "XGBoost",
        "120 sessions flattened; depth selected per fold",
    ),
    "ARIMA": (
        "artifacts/arima_walk_forward/"
        "20260919T174348_864117Z",
        "ARIMA",
        "Close only; order selected per fold",
    ),
    "GRU_w60": (
        "artifacts/gru_walk_forward/"
        "20260919T155453_690801Z",
        "Six_feature_GRU",
        "60 sessions; earlier, larger training sample set",
    ),
}


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def read_predictions(run_path):
    path = PROJECT_ROOT / run_path / "all_predictions.csv"
    table = pd.read_csv(
        path,
        parse_dates=["Forecast_Date", "Target_Date"],
    )

    require(
        len(table) == 756
        and table["Target_Date"].notna().all()
        and table["Target_Date"].is_unique
        and table["Target_Date"].is_monotonic_increasing,
        f"Invalid scoring dates in {path}",
    )
    require(
        table["Target_Date"].dt.year.isin([2020, 2021, 2022]).all(),
        f"Unexpected scoring period in {path}",
    )

    return table.set_index("Target_Date")


def main():
    reference_path = RUNS["LSTM_w120"][0]
    reference = read_predictions(reference_path)

    combined = reference[
        ["Forecast_Date", "Actual_Close", "Naive_Prediction"]
    ].copy()
    combined = combined.rename(columns={"Naive_Prediction": "Naive"})

    provenance = [{
        "Model": "Naive",
        "Source_Run": reference_path,
        "Protocol": "Next Close equals forecast-date Close",
    }]

    for name, (run_path, column, protocol) in RUNS.items():
        table = read_predictions(run_path)

        pd.testing.assert_index_equal(
            table.index,
            reference.index,
        )
        pd.testing.assert_series_equal(
            table["Forecast_Date"],
            reference["Forecast_Date"],
        )

        for price_column in ["Actual_Close", "Naive_Prediction"]:
            np.testing.assert_allclose(
                table[price_column],
                reference[price_column],
                rtol=1e-8,
                atol=1e-8,
            )

        predicted = table[column].to_numpy(dtype=float)
        require(
            np.isfinite(predicted).all(),
            f"{name}: nonfinite predictions.",
        )

        combined[name] = predicted
        provenance.append({
            "Model": name,
            "Source_Run": run_path,
            "Protocol": protocol,
        })
        logger.info("%s: date and actual-price alignment PASS", name)

    models = ["Naive", *RUNS]
    actual = combined["Actual_Close"].to_numpy(dtype=float)
    naive_metrics = regression_metrics(actual, combined["Naive"])

    pooled_rows = []
    yearly_rows = []

    for name in models:
        predicted = combined[name].to_numpy(dtype=float)
        metrics = regression_metrics(actual, predicted)

        pooled_rows.append({
            "Model": name,
            "Samples": len(actual),
            **metrics,
            "Mean_Error": float(np.mean(predicted - actual)),
            "MAE_Improvement_Percent": (
                100 * (1 - metrics["MAE"] / naive_metrics["MAE"])
            ),
            "RMSE_Improvement_Percent": (
                100 * (1 - metrics["RMSE"] / naive_metrics["RMSE"])
            ),
        })

        for year in [2020, 2021, 2022]:
            subset = combined.loc[combined.index.year == year]
            yearly_rows.append({
                "Year": year,
                "Model": name,
                "Samples": len(subset),
                **regression_metrics(
                    subset["Actual_Close"],
                    subset[name],
                ),
                "Mean_Error": float(
                    (subset[name] - subset["Actual_Close"]).mean()
                ),
            })

    pooled = pd.DataFrame(pooled_rows).sort_values(
        "RMSE", kind="stable"
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = PROJECT_ROOT / "reports" / "model_comparison" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    combined.to_csv(output_dir / "aligned_predictions.csv")
    pooled.to_csv(output_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(
        output_dir / "yearly_metrics.csv", index=False
    )
    pd.DataFrame(provenance).to_csv(
        output_dir / "run_provenance.csv", index=False
    )

    # Each model gets its own panel to avoid overlapping six lines.
    recent = combined.tail(126)
    fig, axes = plt.subplots(
        3, 2, figsize=(15, 11), sharex=True, sharey=True
    )

    for axis, name in zip(axes.flat, models):
        axis.plot(
            recent.index, recent["Actual_Close"],
            color="black", linewidth=1.2, label="Actual",
        )
        axis.plot(
            recent.index, recent[name],
            linewidth=1.0, label="Predicted",
        )
        axis.set_title(name)
        axis.set_ylabel("Close (USD, split-adjusted)")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle(
        "Development forecasts — final 126 scoring sessions"
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        output_dir / "forecast_comparison.png",
        dpi=160,
    )
    plt.close(fig)

    print("\nPooled development comparison:")
    print(
        pooled[
            [
                "Model", "Samples", "MAE", "RMSE",
                "MAE_Improvement_Percent",
                "RMSE_Improvement_Percent",
            ]
        ].to_string(index=False, float_format="%.6f")
    )

    print(
        "\nProtocol note: GRU uses the earlier 60-session setup. "
        "See run_provenance.csv for comparison differences."
    )
    logger.info("Saved consolidated report: %s", output_dir)


if __name__ == "__main__":
    run_logged(main, logger)