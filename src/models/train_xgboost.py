import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.download import PROJECT_ROOT
from src.evaluation.metrics import regression_metrics
from src.logger import get_logger, run_logged
from src.models.train_multivariate import (
    load_partition,
    read_json,
    require,
)


logger = get_logger("models.train_xgboost")

DEPTHS = [2, 3, 4]


def flatten(X):
    return np.ascontiguousarray(X.reshape(len(X), -1))


def dollar_changes(metadata):
    return (
        metadata["Actual_Close"] - metadata["Naive_Prediction"]
    ).to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    experiment = read_json(data_dir / "experiment.json")
    config = experiment["config"]
    features = experiment["features"]
    window = experiment["window_size"]

    require(
        window == 120
        and features == [
            "Close",
            "Return_1",
            "RSI_14",
            "Volatility_20",
            "Relative_Volume_20",
            "Range_Ratio",
        ],
        "Expected the six-feature, 120-session dataset.",
    )
    require(
        experiment["scoring_years"] == [2020, 2021, 2022]
        and experiment["target"] == "next_close_minus_current_close",
        "Unexpected development folds or target.",
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = (
        PROJECT_ROOT
        / config["paths"]["artifacts"]
        / "xgboost_walk_forward"
        / stamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    common_settings = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "early_stopping_rounds": 30,
        "min_child_weight": 5,
        "reg_lambda": 10.0,
        "reg_alpha": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "base_score": 0.0,
        "tree_method": "hist",
        "n_jobs": 1,
        "random_state": config["project"]["random_seed"],
    }

    manifest = {
        "prepared_data": str(data_dir.resolve()),
        "xgboost_version": xgb.__version__,
        "experiment": experiment,
        "candidate_depths": DEPTHS,
        "common_settings": common_settings,
        "target": "unscaled_next_session_dollar_change",
        "selection_metric": "selection_year_RMSE",
        "refit_after_selection": False,
    }
    (run_dir / "experiment.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Flattening is session-major: oldest session first.
    flat_names = [
        f"{feature}_lag{window - 1 - position}"
        for position in range(window)
        for feature in features
    ]
    (run_dir / "flattened_columns.json").write_text(
        json.dumps(flat_names, indent=2), encoding="utf-8"
    )

    fold_rows = []
    selected_rows = []
    prediction_tables = []

    for year in experiment["scoring_years"]:
        source_fold = data_dir / str(year)
        fold_dir = run_dir / str(year)
        fold_dir.mkdir()

        preprocessing = read_json(source_fold / "preprocessing.json")
        target_range = float(preprocessing["target_range"])

        require(
            np.isfinite(target_range) and target_range > 0,
            "Invalid target range.",
        )
        require(
            pd.Timestamp(preprocessing["feature_fit_last_date"]).year
            <= year - 2,
            "Preprocessing extends beyond training.",
        )

        for filename in ["preprocessing.json", "feature_scaler.pkl"]:
            shutil.copy2(source_fold / filename, fold_dir / filename)

        X_train, _, train_meta = load_partition(
            source_fold, "train", window, len(features),
            target_range, year,
        )
        X_select, _, select_meta = load_partition(
            source_fold, "selection", window, len(features),
            target_range, year,
        )

        X_train = flatten(X_train)
        X_select = flatten(X_select)

        y_train = dollar_changes(train_meta)
        y_select = dollar_changes(select_meta)

        logger.info(
            "%s: training=%s, selection=%s",
            year, X_train.shape, X_select.shape,
        )

        candidate_rows = []
        best = None

        for depth in DEPTHS:
            model = xgb.XGBRegressor(
                max_depth=depth,
                **common_settings,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_select, y_select)],
                verbose=False,
            )

            # best_iteration is zero-based; the upper bound is exclusive.
            rounds = int(model.best_iteration) + 1
            corrections = model.predict(
                X_select,
                iteration_range=(0, rounds),
            ).astype(float)

            require(
                np.isfinite(corrections).all(),
                "Nonfinite selection predictions.",
            )

            selection_prices = (
                select_meta["Naive_Prediction"].to_numpy()
                + corrections
            )
            metrics = regression_metrics(
                select_meta["Actual_Close"].to_numpy(),
                selection_prices,
            )

            checkpoint = fold_dir / f"depth_{depth}.json"
            model.save_model(checkpoint)

            history = model.evals_result()["validation_0"]["rmse"]
            pd.DataFrame({
                "Round": np.arange(1, len(history) + 1),
                "Selection_RMSE_USD": history,
            }).to_csv(
                fold_dir / f"depth_{depth}_history.csv",
                index=False,
            )

            restored = xgb.XGBRegressor()
            restored.load_model(checkpoint)

            np.testing.assert_allclose(
                restored.predict(
                    X_select,
                    iteration_range=(0, rounds),
                ),
                corrections,
                rtol=1e-7,
                atol=1e-7,
            )

            candidate_rows.append({
                "Depth": depth,
                "Best_Rounds": rounds,
                "Rounds_Completed": len(history),
                **metrics,
            })

            if best is None or metrics["RMSE"] < best["rmse"]:
                best = {
                    "depth": depth,
                    "rounds": rounds,
                    "rmse": metrics["RMSE"],
                    "checkpoint": checkpoint,
                }

            logger.info(
                "%s: depth=%s, rounds=%s, selection RMSE=%.6f; "
                "reload check PASS",
                year, depth, rounds, metrics["RMSE"],
            )

        pd.DataFrame(candidate_rows).to_csv(
            fold_dir / "selection_candidates.csv", index=False
        )

        selected = {
            "Score_Year": year,
            "Depth": best["depth"],
            "Best_Rounds": best["rounds"],
            "Selection_RMSE": best["rmse"],
        }
        selected_rows.append(selected)

        (fold_dir / "selected_model.json").write_text(
            json.dumps(selected, indent=2), encoding="utf-8"
        )
        shutil.copy2(best["checkpoint"], fold_dir / "best_model.json")

        # Load scoring arrays only after model selection.
        X_score, _, score_meta = load_partition(
            source_fold, "score", window, len(features),
            target_range, year,
        )

        restored = xgb.XGBRegressor()
        restored.load_model(fold_dir / "best_model.json")

        corrections = restored.predict(
            flatten(X_score),
            iteration_range=(0, best["rounds"]),
        ).astype(float)

        require(
            corrections.shape == (len(score_meta),)
            and np.isfinite(corrections).all(),
            "Invalid scoring corrections.",
        )

        today = score_meta["Naive_Prediction"].to_numpy()
        actual = score_meta["Actual_Close"].to_numpy()

        predictions = {
            "Naive": today,
            "Training_mean_change": today + float(y_train.mean()),
            "XGBoost": today + corrections,
        }

        for name, predicted in predictions.items():
            score_meta[name] = predicted
            fold_rows.append({
                "Score_Year": year,
                "Model": name,
                "Samples": len(actual),
                **regression_metrics(actual, predicted),
                "Mean_Error": float(np.mean(predicted - actual)),
            })

        score_meta.to_csv(fold_dir / "predictions.csv", index=False)
        prediction_tables.append(score_meta)

        pd.DataFrame(fold_rows).to_csv(
            run_dir / "fold_metrics.csv", index=False
        )
        pd.DataFrame(selected_rows).to_csv(
            run_dir / "selected_models.csv", index=False
        )

    combined = pd.concat(prediction_tables, ignore_index=True)
    require(
        len(combined) == 756
        and combined["Target_Date"].is_unique,
        "Unexpected pooled scoring dates.",
    )
    combined.to_csv(run_dir / "all_predictions.csv", index=False)

    actual = combined["Actual_Close"].to_numpy()
    pooled_rows = []

    for name in ["Naive", "Training_mean_change", "XGBoost"]:
        predicted = combined[name].to_numpy()
        pooled_rows.append({
            "Model": name,
            "Samples": len(actual),
            **regression_metrics(actual, predicted),
            "Mean_Error": float(np.mean(predicted - actual)),
        })

    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(run_dir / "pooled_metrics.csv", index=False)

    print("\nSelected models:")
    print(pd.DataFrame(selected_rows).to_string(index=False))
    print("\nResults by scoring year:")
    print(pd.DataFrame(fold_rows).to_string(index=False))
    print("\nPooled development results:")
    print(pooled.to_string(index=False))

    logger.info("Saved XGBoost comparison: %s", run_dir)


if __name__ == "__main__":
    run_logged(main, logger)