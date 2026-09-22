import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.download import PROJECT_ROOT
from src.evaluation.metrics import regression_metrics
from src.logger import get_logger, run_logged
from src.models.train_final_neural import (
    FEATURES,
    load_partition,
    read_json,
    require,
    save_json,
)


logger = get_logger("models.train_final_xgboost")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_dir = data_dir.resolve()

    experiment = read_json(data_dir / "experiment.json")
    preprocessing = read_json(data_dir / "preprocessing.json")
    config = experiment["config"]

    require(
        experiment["features"] == FEATURES
        and experiment["window_size"] == 120
        and experiment["horizon"] == 1
        and experiment["target"] == "future_close_minus_current_close",
        "Expected the final h1_w120 dataset.",
    )
    require(
        pd.Timestamp(preprocessing["feature_fit_last_date"])
        <= pd.Timestamp("2019-12-31")
        and pd.Timestamp(preprocessing["target_fit_last_date"])
        <= pd.Timestamp("2019-12-31"),
        "Preprocessing extends beyond training.",
    )

    target_range = float(preprocessing["target_range"])
    require(
        np.isfinite(target_range) and target_range > 0,
        "Invalid target range.",
    )

    X_train, _, train_meta = load_partition(
        data_dir, "train", 120, target_range
    )
    X_val, _, val_meta = load_partition(
        data_dir, "validation", 120, target_range
    )

    # Flatten oldest session first, retaining feature order within sessions.
    X_train = X_train.reshape(len(X_train), -1)
    X_val = X_val.reshape(len(X_val), -1)

    require(
        X_train.shape[1] == 720 and X_val.shape[1] == 720,
        "Expected 720 flattened input columns.",
    )

    # Dollar changes, not the scaled neural-network targets.
    y_train = (
        train_meta["Actual_Close"] - train_meta["Naive_Prediction"]
    ).to_numpy(dtype=float)

    y_val = (
        val_meta["Actual_Close"] - val_meta["Naive_Prediction"]
    ).to_numpy(dtype=float)

    actual = val_meta["Actual_Close"].to_numpy(dtype=float)
    today = val_meta["Naive_Prediction"].to_numpy(dtype=float)

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
    depths = [2, 3, 4]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = (
        PROJECT_ROOT / config["paths"]["artifacts"]
        / "final_xgboost" / stamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    for filename in (
        "experiment.json", "preprocessing.json", "feature_scaler.pkl"
    ):
        shutil.copy2(data_dir / filename, run_dir / filename)

    save_json(run_dir / "run_info.json", {
        "prepared_data": str(data_dir),
        "xgboost_version": xgb.__version__,
        "common_settings": common_settings,
        "candidate_depths": depths,
        "model_output": "dollar_correction",
        "flatten_order": "C: oldest session first, features within session",
        "selection": "minimum validation price RMSE",
        "refit_on_validation": False,
        "test_evaluated": False,
    })

    candidate_rows = []
    candidate_predictions = {}

    logger.info(
        "Training shape: %s; validation shape: %s",
        X_train.shape, X_val.shape,
    )

    for depth in depths:
        model = xgb.XGBRegressor(
            max_depth=depth, **common_settings
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        rounds = int(model.best_iteration) + 1
        correction = model.predict(
            X_val, iteration_range=(0, rounds)
        ).astype(float)

        require(
            correction.shape == (len(actual),)
            and np.isfinite(correction).all(),
            f"Depth {depth}: invalid predictions.",
        )

        checkpoint = run_dir / f"depth_{depth}.json"
        model.save_model(checkpoint)

        restored = xgb.XGBRegressor()
        restored.load_model(checkpoint)
        reloaded = restored.predict(
            X_val, iteration_range=(0, rounds)
        ).astype(float)

        require(
            np.allclose(correction, reloaded, rtol=1e-7, atol=1e-8),
            f"Depth {depth}: checkpoint reload check failed.",
        )

        predicted = today + correction
        metrics = regression_metrics(actual, predicted)
        candidate_predictions[depth] = predicted

        history = pd.DataFrame({
            "Round": np.arange(
                1, len(model.evals_result()["validation_0"]["rmse"]) + 1
            ),
            "Validation_RMSE": model.evals_result()["validation_0"]["rmse"],
        })
        history.to_csv(
            run_dir / f"depth_{depth}_history.csv", index=False
        )

        candidate_rows.append({
            "Depth": depth,
            "Best_Rounds": rounds,
            **metrics,
        })

        # Save progress after each successful candidate.
        pd.DataFrame(candidate_rows).to_csv(
            run_dir / "candidate_metrics.csv", index=False
        )
        logger.info(
            "Depth=%s, rounds=%s, validation RMSE=%.6f; reload PASS",
            depth, rounds, metrics["RMSE"],
        )

    # In an exact RMSE tie, prefer the shallower model.
    best = min(candidate_rows, key=lambda row: (row["RMSE"], row["Depth"]))
    depth = best["Depth"]
    rounds = best["Best_Rounds"]

    selected_path = run_dir / "selected.json"
    shutil.copy2(run_dir / f"depth_{depth}.json", selected_path)

    save_json(run_dir / "selected_model.json", {
        "model": "XGBoost_h1_w120",
        "checkpoint": "selected.json",
        "checkpoint_sha256": hashlib.sha256(
            selected_path.read_bytes()
        ).hexdigest(),
        "depth": depth,
        "best_rounds": rounds,
        "prediction_iteration_range": [0, rounds],
        "validation_rmse": best["RMSE"],
        "model_output": "dollar_correction",
        "features": FEATURES,
        "window": 120,
        "horizon": 1,
    })

    predictions = {
        "Naive": today,
        "Training_mean_change": today + float(y_train.mean()),
        "XGBoost_h1_w120": candidate_predictions[depth],
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
    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(run_dir / "validation_metrics.csv", index=False)

    print("\nCandidate results:")
    print(pd.DataFrame(candidate_rows).to_string(
        index=False, float_format="%.6f"
    ))
    print(f"\nSelected depth: {depth}; boosting rounds: {rounds}")
    print("\nValidation comparison:")
    print(metrics_table.to_string(index=False, float_format="%.6f"))

    logger.info("Saved final XGBoost model: %s", run_dir)
    logger.info("Final-test data has not been evaluated")


if __name__ == "__main__":
    run_logged(main, logger)